import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter
import numpy as np

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out
        

class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        #for debuging
        self.output_dim = output_dim
        self.num_prototypes = num_prototypes

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


# Change: Added new class (below) to be used as a substitute for 
# class Prototypes could be used for self.head_lab in rKD.py
class PrototypesHead(nn.Module):

    def __init__(self, num_classes, num_prototypes, num_descriptive, use_last_layer, use_thresh,
                 prototype_activation_function, num_extra_prototypes=0, correct_class_connection=1):
        super().__init__()
        self.epsilon = 1e-4
        self.num_classes = num_classes
        
        self.num_extra_prototypes = num_extra_prototypes
        self.num_prototypes = num_prototypes
        self.correct_class_connection = correct_class_connection
        
        self.num_descriptive = num_descriptive
        self.use_last_layer = use_last_layer
        self.use_thresh = use_thresh
        self.prototype_activation_function = prototype_activation_function


        if self.use_thresh:
            self.alfa = Parameter(torch.Tensor(1, num_classes, num_descriptive))
            nn.init.xavier_normal_(self.alfa, gain=1.0)
        else:
            self.alfa = 1
            self.beta = 0

        self.proto_presence = torch.zeros(num_classes, self.num_prototypes + self.num_extra_prototypes, num_descriptive)  # [c, p, n]

        self.proto_presence = Parameter(self.proto_presence, requires_grad=True)
        nn.init.xavier_normal_(self.proto_presence, gain=1.0)

        self.use_last_layer = use_last_layer
        if self.use_last_layer:
            self.prototype_class_identity = torch.zeros(self.num_descriptive * self.num_classes, self.num_classes)

            for j in range(self.num_descriptive * self.num_classes):
                self.prototype_class_identity[j, j // self.num_descriptive] = 1
            self.last_layer = nn.Linear(self.num_descriptive * self.num_classes, self.num_classes, bias=False)
            positive_one_weights_locations = torch.t(self.prototype_class_identity)
            negative_one_weights_locations = 1 - positive_one_weights_locations

            correct_class_connection = self.correct_class_connection
            incorrect_class_connection = 0 # -0.5
            self.last_layer.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations)
        else:
            self.last_layer = nn.Identity()
    
    def _mix_l2_convolution(self, distances, proto_presence):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        # distances [b, p]
        # proto_presence [c, p, n]
        mixed_distances = torch.einsum('bp,cpn->bcn', distances, proto_presence)

        return mixed_distances  # [b, c, n]
    
    def distance_2_similarity(self, distances):  # [b,c,n]
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            if self.use_thresh:
                distances = distances  # * torch.exp(self.alfa)  # [b, c, n]
            return 1 / (distances + 1)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.last_layer.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.last_layer.weight.copy_(w)

    def forward(self, avg_dist, min_distances, gumbel_scale):
        
        if gumbel_scale == 0:
            proto_presence = torch.softmax(self.proto_presence, dim=1)
        else:
            proto_presence = gumbel_softmax(self.proto_presence * gumbel_scale, dim=1, tau=0.5)

        min_mixed_distances = self._mix_l2_convolution(min_distances, proto_presence)  # [b, c, n]
        avg_mixed_distances = self._mix_l2_convolution(avg_dist, proto_presence)  # [b, c, n]
        x = self.distance_2_similarity(min_mixed_distances)  # [b, c, n]
        x_avg = self.distance_2_similarity(avg_mixed_distances)  # [b, c, n]
        x = x - x_avg
        if self.use_last_layer:
            x = self.last_layer(x.flatten(start_dim=1))
        else:
            x = x.sum(dim=-1)

        # CHANGE: commented out because rKD.py code requires that
        return x, min_distances, proto_presence  # [b,c,n] [b, p] [c, p, n]

    def get_map_class_to_prototypes(self):
        pp = gumbel_softmax(self.proto_presence * 10e3, dim=1, tau=0.5).detach()
        return np.argmax(pp.cpu().numpy(), axis=1)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(self, num_unlabeled_classes, num_prototypes, num_descriptive,
        num_heads, num_extra_prototypes=0, correct_class_connection=1,
        normalize_last_layers=False):

        super().__init__()
        self.num_unlabeled_classes = num_unlabeled_classes
        self.num_prototypes = num_prototypes
        self.num_descriptive = num_descriptive
        self.num_heads = num_heads
        self.num_extra_prototypes = num_extra_prototypes
        self.correct_class_connection = correct_class_connection
        self.normalize_last_layers = normalize_last_layers

        # TODO: maybe we should use MLP:
        # projectors
        # self.projectors = torch.nn.ModuleList(
        #     [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        # )

        # prototypes
        # TODO: change name?:
        self.prototypes = torch.nn.ModuleList(
            [PrototypesHead(num_classes=self.num_unlabeled_classes, num_prototypes=self.num_prototypes,
                            num_descriptive=self.num_descriptive, use_last_layer=True, use_thresh=True, 
                            prototype_activation_function='log', num_extra_prototypes=self.num_extra_prototypes,
                            correct_class_connection=self.correct_class_connection)
                              for _ in range(self.num_heads)])
        
        if self.normalize_last_layers:
            self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, avg_dist, min_distances, gumbel_scale):
        # z = self.projectors[head_idx](feats)
        # z = feats
        # feats = F.normalize(z, dim=1)
        return self.prototypes[head_idx](avg_dist, min_distances, gumbel_scale)

    def forward(self, avg_dist, min_distances, gumbel_scale):
        out = [self.forward_head(h, avg_dist, min_distances, gumbel_scale) \
                for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]

def model_statistics(model):
    total_params = sum(param.numel() for param in model.parameters()) / 1000000.0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
    print('    Total params: %.2fM, trainable params: %.2fM' % (total_params, trainable_params))
