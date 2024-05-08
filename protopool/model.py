import torch
import torch.nn as nn
import torch.nn.functional as F


from protopool.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from protopool.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from protopool.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PrototypeChooser(nn.Module):

    def __init__(self, num_prototypes: int,
                 arch: str = 'resnet34', pretrained: bool = True,
                 add_on_layers_type: str = 'linear',
                 proto_depth: int = 128, inat: bool = False,
                 num_extra_prototypes=0) -> None:
        
        super().__init__()
        self.num_extra_prototypes = num_extra_prototypes
        self.num_prototypes = num_prototypes
        self.proto_depth = proto_depth
        self.prototype_shape = (self.num_prototypes+self.num_extra_prototypes, self.proto_depth, 1, 1)
        self.arch = arch
        self.pretrained = pretrained
        self.inat = inat

        if self.inat:
            self.features = base_architecture_to_features['resnet50'](pretrained=pretrained, inat=True)
        else:
            self.features = base_architecture_to_features[self.arch](pretrained=pretrained)

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in self.features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            raise NotImplementedError
        else:
            add_on_layers = [
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1],
                          kernel_size=1),
                # nn.ReLU(),
                # nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid(),
            ]

            self.add_on_layers = nn.Sequential(*add_on_layers)

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # initial weights
        for m in self.add_on_layers.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def fine_tune_last_only(self):
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.add_on_layers.parameters():
            p.requires_grad = False
        self.prototype_vectors.requires_grad = False
        self.proto_presence.requires_grad = False
        for p in self.last_layer.parameters():
            p.requires_grad = True

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    def forward(self, x, mask_lab): 

        distances = self.prototype_distances(x, mask_lab)  # [b, C, H, W] -> [b, p, h, w]

        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])).squeeze()  # [b, p]
        avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                        distances.size()[3])).squeeze()  # [b, p]

        return avg_dist, min_distances, distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))

        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        intermediate_result = - 2 * xp + p2_reshape 

        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _calculate_prototype_distances(self, x, prototypes):
        '''
        Calculates the L2 distances between the input x and the given prototypes.
        '''
        x2 = x ** 2
        ones = torch.ones_like(prototypes, device=x.device)
        x2_patch_sum = F.conv2d(input=x2, weight=ones)

        p2 = prototypes ** 2
        p2_sum = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2_sum.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototypes)
        intermediate_result = -2 * xp + p2_reshape 

        distances = F.relu(x2_patch_sum + intermediate_result)
        return distances    
   
    def _l2_convolution_mask(self, x, mask_lab):
        '''
        Apply self.prototype_vectors as l2-convolution filters on input x.
        The first 202 prototypes are applied to all images.
        The last 20 prototypes are applied only to images not masked by mask_lab.
        '''
        if mask_lab is None:    # when old_model and test_loader
            distances = self._calculate_prototype_distances(x, self.prototype_vectors)
            return distances
        else:
            distances_general = self.prototype_vectors[:self.num_prototypes, :, :, :]
            distances_general = self._calculate_prototype_distances(x, distances_general)

        if self.num_extra_prototypes > 0: # and x[~mask_lab].size(0) != 0:
            
            # x_selected = x[~mask_lab]
            x_selected = x
                
            prototypes_specific = self.prototype_vectors[-self.num_extra_prototypes:, :, :, :]
            distances_specific = self._calculate_prototype_distances(x_selected, prototypes_specific)
            
            # distances_specific_full = torch.zeros((x.shape[0], self.num_extra_prototypes, x.shape[2], x.shape[3]), device=x.device)
            # distances_specific_full[~mask_lab] = distances_specific
            
            distances_specific_full = distances_specific

            distances = torch.cat([distances_general, distances_specific_full], dim=1)
        else:
            distances = distances_general

        return distances

    def prototype_distances(self, x, mask_lab):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        distances = self._l2_convolution_mask(conv_features, mask_lab)  # [b, p, h, w]
        return distances  # [b, n, h, w], [b, p, h, w]

    def __repr__(self):
        res = super(PrototypeChooser, self).__repr__()
        return res
