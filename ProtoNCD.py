import os
import random
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import warnings

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import contextlib
from model.model_utils import *
from model.losses import SinkhornKnopp, StableSinkhornKnopp, KD, SinkhornKnopp_2
from model.get_model import get_backbone
from utils.parse import get_args
from utils.other import validate_difference
from data.augmentations import get_transform
from data.get_datasets import get_datasets
from utils.eval_utils import split_cluster_acc_v2, cluster_eval
import wandb
import numpy as np
from tqdm import tqdm
import copy
from torch.optim import AdamW, SGD
import pickle as pkl
from protopool.model import PrototypeChooser
from protopool.utils import mixup_data, find_high_activation_crop, compute_rf_prototype, compute_proto_layer_rf_info_v2

# protopool:
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
import sys
import heapq

from scipy.optimize import linear_sum_assignment


class Net(nn.Module):

    def __init__(self,
                 backbone,
                 num_labeled_classes,
                 num_unlabeled_classes,
                 num_prototypes,
                 num_descriptive,
                 num_heads,
                 gumbel_max_lab=False,
                 num_extra_prototypes=0,
                 correct_class_connection=1,
                 normalize_last_layers=False):
        
        super().__init__()

        self.encoder = backbone
        self.num_labeled_classes = num_labeled_classes
        self.num_unlabeled_classes = num_unlabeled_classes
        self.num_prototypes = num_prototypes
        self.num_descriptive = num_descriptive
        self.num_heads = num_heads
        self.gumbel_max_lab = gumbel_max_lab
        self.num_extra_prototypes = num_extra_prototypes
        self.correct_class_connection = correct_class_connection
        self.normalize_last_layers = normalize_last_layers
        
        self.head_lab = PrototypesHead(num_classes=self.num_labeled_classes, num_prototypes=self.num_prototypes,
                                        num_descriptive=self.num_descriptive, use_last_layer=True,
                                        use_thresh=True, prototype_activation_function='log')

        if num_heads > 0: 
            self.head_unlab = MultiHead(num_unlabeled_classes=self.num_unlabeled_classes, num_prototypes=self.num_prototypes,
                                        num_descriptive=self.num_descriptive, num_heads=self.num_heads,
                                        num_extra_prototypes=self.num_extra_prototypes,
                                        correct_class_connection=self.correct_class_connection,
                                        normalize_last_layers=self.normalize_last_layers)

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()

    def forward_heads(self, avg_dist, min_distances, distances, gumbel_scale):

        if self.gumbel_max_lab:
            gumbel_scale_lab = 10e3
        else:
            gumbel_scale_lab = gumbel_scale

        if self.num_extra_prototypes:
            x_lab, min_distances_lab, proto_presence_lab = self.head_lab(avg_dist[:, :self.num_prototypes], min_distances[:, :self.num_prototypes], gumbel_scale_lab)
        else:
            x_lab, min_distances_lab, proto_presence_lab = self.head_lab(avg_dist, min_distances, gumbel_scale_lab)

        out = {"logits_lab": x_lab, 
                "min_distances_lab": min_distances_lab,
                "proto_presence_lab": proto_presence_lab,
                "distances": distances}
                
        if hasattr(self, "head_unlab"):
            x_unlab, min_distances_unlab, proto_presence_unlab = self.head_unlab(avg_dist, min_distances, gumbel_scale)
            out.update({
                "logits_unlab": x_unlab, 
                "min_distances_unlab": min_distances_unlab,
                "proto_presence_unlab": proto_presence_unlab})
        return out

    def forward(self, views, mask_lab, gumbel_scale):
        if isinstance(views, list):
            feats = [self.encoder(view, mask_lab) for view in views]

            out = [self.forward_heads(avg_dist, min_distances, distances, gumbel_scale) for avg_dist, min_distances, distances in feats]

            out_dict = dict()
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:

            avg_dist, min_distances, distances = self.encoder(views, mask_lab)
            out = self.forward_heads(avg_dist, min_distances, distances, gumbel_scale)

            return out

def lambda1(epoch): 
    start_val = 1.3
    end_val = 10 ** 3
    epoch_interval = args.gumbel_time
    alpha = (end_val / start_val) ** 2 / epoch_interval
    
    return start_val * np.sqrt(alpha *
                    (epoch)) if epoch < epoch_interval else end_val


def dist_loss(model, min_distances, proto_presence, top_k, sep=False):
    #         model, [b, p],        [b, p, n],      [scalar]
    max_dist = (model.encoder.prototype_shape[1]
                * model.encoder.prototype_shape[2]
                * model.encoder.prototype_shape[3])
    basic_proto = proto_presence.sum(dim=-1).detach()  # [b, p]
    _, idx = torch.topk(basic_proto, top_k, dim=1)  # [b, n]
    binarized_top_k = torch.zeros_like(basic_proto)
    binarized_top_k.scatter_(1, src=torch.ones_like(
        basic_proto), index=idx)  # [b, p]
    inverted_distances, _ = torch.max(
        (max_dist - min_distances) * binarized_top_k, dim=1)  # [b]
    cost = torch.mean(max_dist - inverted_distances)
    return cost


def adjust_learning_rate(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= rate


def save_model(model, path, epoch):
    torch.save({
        'state_dict': model.state_dict(),
        'epoch': epoch
    }, path)


def train_pretrain(train_loader, train_push_loader, test_loader, args):

    backbone = PrototypeChooser(num_prototypes=args.num_prototypes,
                 arch='resnet50', pretrained=True, add_on_layers_type='log',
                 proto_depth=256, inat=True, num_extra_prototypes=args.num_extra_prototypes)
    
    model = Net(backbone,
                num_labeled_classes=args.num_labeled_classes,
                num_unlabeled_classes=args.num_unlabeled_classes,
                num_prototypes=args.num_prototypes,
                num_descriptive=args.num_descriptive,
                num_heads=0)

    model = model.cuda()
    model_statistics(model)

    if args.warmup:
        model.encoder.features.requires_grad_(False)
        model.head_lab.last_layer.requires_grad_(True)

    clst_weight = 0.8
    sep_weight = -0.08
    tau = 1

    warm_optimizer = torch.optim.Adam(
        [{'params': model.encoder.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.head_lab.proto_presence, 'lr': 3 * args.lr},
         {'params': model.encoder.prototype_vectors, 'lr': 3 * args.lr}])
    joint_optimizer = torch.optim.Adam(
        [{'params': model.encoder.features.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3},
         {'params': model.encoder.add_on_layers.parameters(), 'lr': 3 * args.lr,
          'weight_decay': 1e-3},
         {'params': model.head_lab.proto_presence, 'lr': 3 * args.lr},
         {'params': model.encoder.prototype_vectors, 'lr': 3 * args.lr}]
    )
    push_optimizer = torch.optim.Adam(
        [{'params': model.head_lab.last_layer.parameters(), 'lr': args.lr / 10,
          'weight_decay': 1e-3}, ]
    )
    optimizer = warm_optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize wandb run
    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(project=args.project,
               entity=args.entity,
               config=state,
               name=f'{args.time_run}_{args.comment}_protopool_lr_{args.lr}',
               dir=args.log_dir)

    min_val_loss = np.Inf
    max_val_tst = 0
    epochs_no_improve = 0
    
    if not args.only_push:
        # train
        steps = False
        for epoch in range(args.pretrain_epochs):
            gumbel_scalar = lambda1(epoch) if args.pp_gumbel else 0

            if args.warmup and args.warmup_time == epoch:
                model.encoder.features.requires_grad_(True)
                optimizer = joint_optimizer            
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.1)
                steps = True
                print("Warm up ends")
            
            trn_loss = 0
            model.train()
            bar = tqdm(train_loader)
            for i, batch in enumerate(bar):
                data, label, _ = batch
                label_p = label.numpy().tolist()

                label = label.cuda(non_blocking=True)
                data = data.cuda(non_blocking=True)

                if args.mixup_data:
                    data, targets_a, targets_b, lam, _ = mixup_data(
                        data, label, 0.5)

                outputs = model([data], mask_lab=None, gumbel_scale=gumbel_scalar)

                prob = outputs['logits_lab'][0]
                min_distances = outputs['min_distances_lab'][0]
                proto_presence = outputs['proto_presence_lab'][0]

                if args.mixup_data:
                    entropy_loss = lam * \
                        criterion(prob, targets_a) + (1 - lam) * \
                        criterion(prob, targets_b)
                else:
                    entropy_loss = criterion(prob, label)
                
                orthogonal_loss = torch.Tensor([0]).cuda()
                if args.pp_ortho:
                    for c in range(0, model.head_lab.proto_presence.shape[0], 1000):
                        orthogonal_loss_p = \
                            torch.nn.functional.cosine_similarity(model.head_lab.proto_presence.unsqueeze(2)[c:c+1000],
                                                                    model.head_lab.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                        orthogonal_loss += orthogonal_loss_p
                    orthogonal_loss = orthogonal_loss / (args.num_descriptive * args.num_labeled_classes) - 1

                proto_presence = proto_presence[label_p]
                inverted_proto_presence = 1 - proto_presence
                clst_loss_val = \
                    dist_loss(model, min_distances, proto_presence,
                                args.num_descriptive)  
                sep_loss_val = dist_loss(model, min_distances, inverted_proto_presence,
                                            args.num_prototypes - args.num_descriptive)  

                prototypes_of_correct_class = proto_presence.sum(
                    dim=-1).detach()
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                            dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                l1_mask = 1 - \
                    torch.t(model.head_lab.prototype_class_identity).cuda()
                l1 = (model.head_lab.last_layer.weight * l1_mask).norm(p=1)

                loss = entropy_loss + clst_loss_val * clst_weight + \
                    sep_loss_val * sep_weight + 1e-4 * l1 + orthogonal_loss 

                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({"train/loss": loss,
                        "train/entropy": entropy_loss.item(),
                        "train/clst": clst_loss_val.item(),
                        "train/sep": sep_loss_val.item(),
                        "train/l1": l1.item(),
                        "train/avg_sep": avg_separation_cost.item(),
                        "train/orthogonal_loss": orthogonal_loss.item()})
                        # step=epoch*len(train_loader) + i)
                
                bar.set_postfix(
                    {"loss": "{:.2f}".format(loss.detach().cpu().numpy()[0])})
            
                trn_loss += loss.item()

                if args.debugging:
                    break

            trn_loss /= len(train_loader)

            if steps:
                lr_scheduler.step()

            model.eval()
            tst_loss = np.zeros((args.num_labeled_classes, 1))
            prob_leaves = np.zeros((args.num_labeled_classes, 1))
            tst_acc, total = 0, 0
            total_entropy_loss, total_clst_loss, total_sep_loss, total_orthogonal_loss, total_l1 = 0, 0, 0, 0, 0
            tst_tqdm = enumerate(test_loader, 0)
            with torch.no_grad():
                for i, (data, label, _) in tst_tqdm:
                    data = data.cuda()
                    label_p = label.detach().numpy().tolist()
                    label = label.cuda()

                    # ===================forward=====================

                    outputs = model([data], mask_lab=None, gumbel_scale=gumbel_scalar)

                    prob = outputs['logits_lab'][0]
                    min_distances = outputs['min_distances_lab'][0]
                    proto_presence = outputs['proto_presence_lab'][0]

                    loss = criterion(prob, label)
                    entropy_loss = loss

                    orthogonal_loss = 0
                    orthogonal_loss = torch.Tensor([0]).cuda()                                                                                                                                            
                    if args.pp_ortho: 
                        for c in range(0, model.head_lab.proto_presence.shape[0], 1000):
                            orthogonal_loss_p = \
                                torch.nn.functional.cosine_similarity(model.head_lab.proto_presence.unsqueeze(2)[c:c+1000],
                                                                        model.head_lab.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            orthogonal_loss += orthogonal_loss_p
                        orthogonal_loss = orthogonal_loss / (args.num_descriptive * args.num_labeled_classes) - 1
                    inverted_proto_presence = 1 - proto_presence

                    l1_mask = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
                    l1 = (model.head_lab.last_layer.weight * l1_mask).norm(p=1)

                    proto_presence = proto_presence[label_p]
                    inverted_proto_presence = inverted_proto_presence[label_p]
                    clst_loss_val = dist_loss(model, min_distances, proto_presence, args.num_descriptive) * clst_weight
                    sep_loss_val = dist_loss(model, min_distances, inverted_proto_presence, args.num_prototypes - args.num_descriptive, sep=True) * sep_weight
                    loss = entropy_loss + clst_loss_val + sep_loss_val + orthogonal_loss + 1e-4 * l1
                    tst_loss += loss.item()

                    _, predicted = torch.max(prob, 1)
                    prob_leaves += prob.mean(dim=0).unsqueeze(
                        1).detach().cpu().numpy()
                    true = label
                    tst_acc += (predicted == true).sum()
                    total += label.size(0)

                    # Accumulate losses
                    total_entropy_loss += entropy_loss.item()
                    total_clst_loss += clst_loss_val.item()
                    total_sep_loss += sep_loss_val.item()
                    total_orthogonal_loss += orthogonal_loss.item()
                    total_l1 += l1.item()
                    
                    if args.debugging:
                        break

            tst_loss /= len(test_loader)
            tst_acc = tst_acc.item() / total        
            
            wandb.log({"test/acc": tst_acc,
                    "test/loss": tst_loss.mean(),
                    "test/entropy": total_entropy_loss / len(test_loader),
                    "test/clst": total_clst_loss / len(test_loader),
                    "test/sep": total_sep_loss / len(test_loader),
                    "test/orthogonal_loss": total_orthogonal_loss / len(test_loader),
                    "test/l1": total_l1 / len(test_loader)})
                    # step=epoch)

            print(f'Epoch {epoch}|{args.pretrain_epochs}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
                    f'| acc: {tst_acc:.5f}, orthogonal: {total_orthogonal_loss / len(test_loader):.5f} '
                    f'(minimal test-loss: {min_val_loss:.5f}, early stop: {epochs_no_improve}|{args.earlyStopping}) - ')

            if args.save_model_every and epoch % args.save_model_every == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                state_save = {
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state_save,
                    os.path.join(args.model_save_dir, f"pretrained_checkpoint_epoch_{epoch}.pth"))

            if args.save_model:
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(
                        args.model_save_dir, "pretrained_{}_{}_{}.pth".format(
                            args.dataset, args.num_labeled_classes,
                            args.num_unlabeled_classes)))

            if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
                # save the best model
                if tst_acc > max_val_tst:
                    save_model(model, os.path.join(args.model_save_dir, 'best_model.pth'), epoch)

                epochs_no_improve = 0
                if tst_loss.mean() < min_val_loss:
                    min_val_loss = tst_loss.mean()
                if tst_acc > max_val_tst:
                    max_val_tst = tst_acc
            else:
                epochs_no_improve += 1

            if args.use_scheduler:
                # scheduler.step()
                if epochs_no_improve > 5:
                    adjust_learning_rate(optimizer, 0.95)

            if args.earlyStopping is not None and epochs_no_improve > args.earlyStopping:
                print('\033[1;31mEarly stopping!\033[0m')
                break
            
            if args.debugging:
                break

    ####################################
    #            push step             #
    ####################################
    print('Model push')

    if args.only_push:
        # load best model with path:
        print('Loading best model')
        state_dict = torch.load(args.only_push_model_path)
        model.load_state_dict(state_dict['state_dict'])
        model = model.cuda()

    push_folder = os.path.join(args.model_save_dir, args.proto_img_dir)

    model.eval()
    ####################################
    #          validation step         #
    ####################################    

    tst_loss = np.zeros((args.num_labeled_classes, 1))
    tst_acc, total = 0, 0
    tst_tqdm = enumerate(test_loader, 0)
    with torch.no_grad():
        for i, (data, label, _) in tst_tqdm:
            data = data.cuda()
            label = label.cuda()

            # ===================forward=====================
            outputs = model([data], mask_lab=None, gumbel_scale=10e3)

            prob = outputs['logits_lab'][0]

            loss = criterion(prob, label)
            entropy_loss = loss

            l1_mask = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
            l1 = 1e-4 * (model.head_lab.last_layer.weight * l1_mask).norm(p=1)

            loss = entropy_loss + l1
            tst_loss += loss.item()

            _, predicted = torch.max(prob, 1)
            true = label
            tst_acc += (predicted == true).sum()
            total += label.size(0)

        tst_loss /= len(test_loader)
        tst_acc = tst_acc.item() / total
    print(
        f'Before tuning, test loss: {tst_loss.mean():.5f} | acc: {tst_acc:.5f}')

    global_min_proto_dist = np.full(model.head_lab.num_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [model.head_lab.num_prototypes,
         model.encoder.prototype_shape[1],
         model.encoder.prototype_shape[2],
         model.encoder.prototype_shape[3]])

    proto_rf_boxes = np.full(shape=[model.head_lab.num_prototypes, 6],
                                fill_value=-1)
    proto_bound_boxes = np.full(shape=[model.head_lab.num_prototypes, 6],
                                        fill_value=-1)

    search_batch_size = train_push_loader.batch_size     

    for push_iter, (search_batch_input, search_y, _) in enumerate(train_push_loader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''

        start_index_of_search_batch = push_iter * search_batch_size

        # TODO: uncomment the following lines:

        update_prototypes_on_batch(search_batch_input=search_batch_input, 
                                   start_index_of_search_batch=start_index_of_search_batch,
                                   model=model,
                                   global_min_proto_dist=global_min_proto_dist,
                                   global_min_fmap_patches=global_min_fmap_patches,
                                   proto_rf_boxes=proto_rf_boxes,
                                   proto_bound_boxes=proto_bound_boxes,
                                   class_specific=True,
                                   search_y=search_y,
                                   prototype_layer_stride=1,
                                   dir_for_saving_prototypes=push_folder,
                                   prototype_img_filename_prefix='prototype-img',
                                   prototype_self_act_filename_prefix='prototype-self-act',
                                   prototype_activation_function_in_numpy=None,
                                   h=None)

    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(model.encoder.prototype_shape))
    model.encoder.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())


    save_model(model, os.path.join(args.model_save_dir, 'model_just_after_push.pth'), 0)

    print('Fine-tuning')
    max_val_tst = 0
    min_val_loss = 10e5
    for tune_epoch in range(0, args.pretrain_tune_epochs):
        trn_loss = 0
        trn_tqdm = enumerate(train_loader, 0)
        model.train()
        for i, (data, label, _) in trn_tqdm:
            data = data.cuda()
            label = label.cuda()

            # ===================forward=====================
            if args.mixup_data:
                data, targets_a, targets_b, lam, _ = mixup_data(data, label, 0.5)

            # ===================forward=====================
            outputs = model([data], mask_lab=None, gumbel_scale=10e3)
            
            prob = outputs['logits_lab'][0]

            if args.mixup_data:
                entropy_loss = lam * \
                    criterion(prob, targets_a) + (1 - lam) * \
                    criterion(prob, targets_b)
            else:
                entropy_loss = criterion(prob, label)

            l1_mask = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
            l1 = 1e-4 * (model.head_lab.last_layer.weight * l1_mask).norm(p=1)

            loss = entropy_loss + l1

            # ===================backward====================
            push_optimizer.zero_grad()
            loss.backward()
            push_optimizer.step()
            trn_loss += loss.item()

            wandb.log({"train_push/loss": loss,
                    "train_push/l1": l1.item()})
                    # step=tune_epoch*len(train_loader) + i)

            if args.debugging:
                break

        ####################################
        #          validation step         #
        ####################################
        model.eval()
        tst_loss = np.zeros((args.num_labeled_classes, 1))
        tst_acc, total = 0, 0
        total_entropy_loss = 0
        tst_tqdm = enumerate(test_loader, 0)
        with torch.no_grad():
            for i, (data, label, _) in tst_tqdm:
                data = data.cuda()
                label = label.cuda()

                # ===================forward=====================
                outputs = model([data], mask_lab=None, gumbel_scale=10e3)

                prob = outputs['logits_lab'][0]

                loss = criterion(prob, label)
                entropy_loss = loss

                l1_mask = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
                l1 = 1e-4 * (model.head_lab.last_layer.weight * l1_mask).norm(p=1)

                loss = entropy_loss + l1
                tst_loss += loss.item()

                _, predicted = torch.max(prob, 1)
                true = label
                tst_acc += (predicted == true).sum()
                total += label.size(0)
                total_entropy_loss += entropy_loss.item()

            tst_loss /= len(test_loader)
            tst_acc = tst_acc.item() / total
        ####################################
        #             logger               #
        ####################################

        wandb.log({"test_push/acc": tst_acc,
                "test_push/loss": tst_loss.mean(),
                "test_push/entropy": total_entropy_loss / len(test_loader),
                "test_push/l1": l1.item()})
                # step=tune_epoch)

        if trn_loss is None:
            trn_loss = loss.mean().detach()
            trn_loss = trn_loss.cpu().numpy() / len(train_loader)
        print(f'Epoch {tune_epoch}|{args.pretrain_tune_epochs}, train loss: {trn_loss:.5f}, test loss: {tst_loss.mean():.5f} '
              f'| acc: {tst_acc:.5f}, (minimal test-loss: {min_val_loss:.5f}- )')

        ####################################
        #  scheduler and early stop step   #
        ####################################
        if (tst_loss.mean() < min_val_loss) or (tst_acc > max_val_tst):
            # save the best model
            if tst_acc > max_val_tst:
                save_model(model, os.path.join(args.model_save_dir, 'best_model_push.pth'), tune_epoch)
            if tst_loss.mean() < min_val_loss:
                min_val_loss = tst_loss.mean()
            if tst_acc > max_val_tst:
                max_val_tst = tst_acc

        if (tune_epoch + 1) % 5 == 0:
            adjust_learning_rate(push_optimizer, 0.95)

        if args.debugging:
            break

    # Finalize the wandb run
    wandb.finish()
    print('Finished training model. Have a nice day :)')


# DISCOVERY TRAINING STAGE:
def add_extra_prototypes_and_load_state_dict(student_model, teacher_state_dict, noise_multiplier=0.001, add_random=False):

    n_of_prototypes_in_student_model = student_model.encoder.prototype_vectors.shape[0]
    n_of_prototypes_in_teacher_model = teacher_state_dict['encoder.prototype_vectors'].shape[0]

    n_to_pad = n_of_prototypes_in_student_model - n_of_prototypes_in_teacher_model

    prototype_vectors_copy = copy.deepcopy(teacher_state_dict['encoder.prototype_vectors'])    

    if not add_random:

        vectors = prototype_vectors_copy.squeeze()  # This will change the shape to [101, 256]
        diffs = vectors.unsqueeze(1) - vectors.unsqueeze(0)  # This will have a shape of [101, 101, 256]
        norms = torch.norm(diffs, dim=2)  # This results in a [101, 101] tensor containing the norms
        # mean, median, max, min, std
        print("Norms of differences in prototype vectors:")
        print("Mean: ", torch.mean(norms).item())
        print("Std: ", torch.std(norms).item())
        print("Median: ", torch.median(norms).item())
        print("Max: ", torch.max(norms).item())
        print("Min: ", torch.min(norms).item())

        noise = torch.randn_like(prototype_vectors_copy) * torch.mean(norms) * noise_multiplier

        prototype_vectors_noisy = (prototype_vectors_copy + noise)

        validate_difference(prototype_vectors_copy, prototype_vectors_noisy)

        p_to_add = prototype_vectors_noisy
    else:
        p_to_add = torch.randn_like(prototype_vectors_copy)

    teacher_state_dict['encoder.prototype_vectors'] = torch.cat((teacher_state_dict['encoder.prototype_vectors'], p_to_add), dim=0)

    padding_ones = torch.ones(n_to_pad, *teacher_state_dict['encoder.prototype_vectors'].size()[1:]).cuda()
    teacher_state_dict['encoder.ones'] = torch.cat((teacher_state_dict['encoder.ones'], padding_ones), dim=0)

    # TODO: please write the code witout strict=False:
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    student_model.load_state_dict(teacher_state_dict, strict=False)
    

def log_gradient_norms(model):
    """
    Logs the gradient norms of each parameter in the model.
    """
    gradient_norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_norm = parameter.grad.norm().item()
            gradient_norms[name] = grad_norm
    return gradient_norms


def cross_entropy_loss(preds, targets, temperature):
    preds = F.log_softmax(preds / temperature, dim=-1)
    return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)


def get_connections_weights(old_model):
    sum_of_correct_class_connections = 0
    for n in range(old_model.head_lab.last_layer.weight.shape[0]):
        sum_of_correct_class_connections += old_model.head_lab.last_layer.weight[n, 
                                            n*old_model.head_lab.num_descriptive:(n+1)*old_model.head_lab.num_descriptive].sum()
    
    average_of_correct_class_connections = \
        sum_of_correct_class_connections / (old_model.head_lab.last_layer.weight.shape[0] * old_model.head_lab.num_descriptive)
    
    return average_of_correct_class_connections.item()


def assign_value_to_correct_class_connections(model, value):
        
    for h in range(model.num_heads):
        model.head_unlab.prototypes[h].prototype_class_identity = torch.zeros(model.head_unlab.prototypes[h].num_descriptive * model.head_unlab.prototypes[h].num_classes, model.head_unlab.prototypes[h].num_classes)

        for j in range(model.head_unlab.prototypes[h].num_descriptive * model.head_unlab.prototypes[h].num_classes):
            model.head_unlab.prototypes[h].prototype_class_identity[j, j // model.head_unlab.prototypes[h].num_descriptive] = 1
        model.head_unlab.prototypes[h].last_layer = nn.Linear(model.head_unlab.prototypes[h].num_descriptive * model.head_unlab.prototypes[h].num_classes, model.head_unlab.prototypes[h].num_classes, bias=False)
        positive_one_weights_locations = torch.t(model.head_unlab.prototypes[h].prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = value
        incorrect_class_connection = 0 # -0.5
        model.head_unlab.prototypes[h].last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)


def train_discover(train_loader, train_push_loader, train_val_loader, test_loader, args):

    print("\n\033[91mNumber of GPUs available: {}\033[00m\n".format(torch.cuda.device_count()))
    
    if args.not_use_kd_loss:
        print("\n\033[91mAblation: Not using kd_loss\033[00m\n")

    old_backbone = PrototypeChooser(num_prototypes=args.num_prototypes, arch='resnet50', pretrained=True,
                                    add_on_layers_type='log', proto_depth=256, inat=True, num_extra_prototypes=0)
    
    old_model = Net(old_backbone,
                    num_labeled_classes=args.num_labeled_classes,
                    num_unlabeled_classes=args.num_unlabeled_classes,
                    num_prototypes=args.num_prototypes,
                    num_descriptive=args.num_descriptive,
                    num_heads=0)
        
    print("Load supervised pretrain from {}".format(args.pretrained_path))
    checkpoint = torch.load(args.pretrained_path)
    if 'model_state_dict' in checkpoint:
        state_dict_to_load = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict_to_load = checkpoint['state_dict']
    old_model.load_state_dict(state_dict_to_load)
        
    backbone = PrototypeChooser(num_prototypes=args.num_prototypes, arch='resnet50', pretrained=True,
                                add_on_layers_type='log', proto_depth=256, inat=True, 
                                num_extra_prototypes=args.num_extra_prototypes)
    
    model = Net(backbone,
                num_labeled_classes=args.num_labeled_classes,
                num_unlabeled_classes=args.num_unlabeled_classes,
                num_prototypes=args.num_prototypes,
                num_descriptive=args.num_descriptive,
                num_heads=args.num_heads,
                gumbel_max_lab=args.gumbel_max_lab,
                num_extra_prototypes=args.num_extra_prototypes,
                # keep it 1 for now here
                correct_class_connection=1,
                # keep it False for now here
                normalize_last_layers=args.normalize_last_layers)

    if args.num_extra_prototypes:
        add_extra_prototypes_and_load_state_dict(model, state_dict_to_load, noise_multiplier=args.noise_multiplier, add_random=args.add_random_prototypes)

    else:
        # TODO: please write the code witout strict=False:
        # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.load_state_dict(state_dict_to_load, strict=False)

    if args.normalize_last_layers:
        model.normalize_prototypes()

    # it has to be after normalize_last_layers
    if args.average_last_layer_weights:
        correct_class_connection = get_connections_weights(model)
        assign_value_to_correct_class_connections(model, correct_class_connection)

    model = model.cuda()
    old_model = old_model.cuda()
    
    old_model.eval()
    for _, p in old_model.named_parameters():
        p.requires_grad = False
          
    if args.freeze_backbone:
        # freeze: features, add_on_layers, prototype_vectors
        for param in model.encoder.parameters():
            param.requires_grad = False

    if args.freeze_backbone_except_num_extra_prototypes:
        for param in model.encoder.parameters():
            param.requires_grad = False
        # the gradient for pretrained prototypes
        # is going to be zero-ed in the training loop
        model.encoder.prototype_vectors.requires_grad_(True)

    if args.freeze_pretrained_part_of_model:
        for name, param in model.named_parameters():
            if not name.startswith('head_unlab'):
                param.requires_grad = False

    if args.freeze_pretrained_slots:
        model.head_lab.proto_presence.requires_grad_(False)

    if args.freeze_encoder:
        for name, param in model.encoder.features.named_parameters():
            param.requires_grad_(False)

    if args.warmup:
        model.encoder.features.requires_grad_(False)
        # TODO: probably not necessary:
        model.head_lab.last_layer.requires_grad_(True)
        for h in range(args.num_heads):
            model.head_unlab.prototypes[h].last_layer.requires_grad_(True)

    model_statistics(model)

    clst_weight = 0.8
    sep_weight = -0.08
    tau = 1

    warm_proto_params = [{'params': model.head_unlab.prototypes[h].proto_presence, 'lr': 3 * args.lr} for h in range(args.num_heads)]
    joint_proto_params= [{'params': model.head_unlab.prototypes[h].proto_presence, 'lr': 3 * args.lr} for h in range(args.num_heads)]
    push_proto_params= [{'params': model.head_unlab.prototypes[h].last_layer.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3} for h in range(args.num_heads)]

    warm_optimizer = torch.optim.Adam(
        # TODO: why in the pre warmup optimizer we do: .encoder.add_on_layers.parameters(), isn't that part frozen?
        [{'params': model.encoder.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.head_lab.proto_presence, 'lr': 3 * args.lr},
         *warm_proto_params,
         {'params': model.encoder.prototype_vectors, 'lr': 3 * args.lr}])
    joint_optimizer = torch.optim.Adam(
        [{'params': model.encoder.features.parameters(), 'lr': args.lr / 10, 'weight_decay': 1e-3},
         {'params': model.encoder.add_on_layers.parameters(), 'lr': 3 * args.lr, 'weight_decay': 1e-3},
         {'params': model.head_lab.proto_presence, 'lr': 3 * args.lr},
         *joint_proto_params,
         {'params': model.encoder.prototype_vectors, 'lr': 3 * args.lr}]
    )
    push_optimizer = torch.optim.Adam(
        [{'params': model.head_lab.last_layer.parameters(), 'lr': args.lr / 10,
          'weight_decay': 1e-3}, 
        *push_proto_params])

    optimizer = warm_optimizer

    if args.restart_from_checkpoint and args.restart_checkpoint_path is not None:
        print("\n\033[91mRestarting training from checkpoint: {}\033[00m\n".format(args.restart_checkpoint_path))
        checkpoint = torch.load(args.restart_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        start_epoch = checkpoint['epoch']
        args.only_push = checkpoint['is_push']
        
        if checkpoint['is_push']:
            optimizer = push_optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])

        # here we have args.warmup_time < start_epoch because we save epoch+1:
        elif args.warmup and args.warmup_time < start_epoch:
            optimizer = joint_optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            optimizer = warm_optimizer
            optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        start_epoch = 0 

    state = {k: v for k, v in args._get_kwargs()}
    wandb.init(
        project=args.project,
        entity=args.entity,
        config=state,
        name=f"{args.time_run}_{args.comment}",
        dir=args.log_dir
    )
    
    nlc = args.num_labeled_classes

    sk = StableSinkhornKnopp()

    # TODO: remove loss_per_head (also below)
    loss_per_head = torch.zeros(args.num_heads).cuda()

    if not args.only_push:
        steps = False

        # TODO: remove best_scores (also below)
        best_scores = {"epoch": 0, "acc": 0}
        for epoch in range(start_epoch, args.max_epochs + args.extra_epochs):
            gumbel_scalar = lambda1(epoch) if args.pp_gumbel else 0
            model.train()
            bar = tqdm(train_loader)

            # TODO: check if could use:
            scaler = GradScaler()
            amp_cm = autocast() if args.amp else contextlib.nullcontext()

            if args.warmup and args.warmup_time == epoch:
                model.encoder.features.requires_grad_(True)
                optimizer = joint_optimizer
                # TODO: keep lr constant?
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.1)
                steps = True
                print("Warm up ends")
            
            trn_loss = 0

            for batch_idx, batch in enumerate(bar):
                images, labels, uq_idxs, mask_lab = batch
                mask_lab = mask_lab[:, 0]
                label_p = labels.numpy().tolist()

                labels = labels.cuda(non_blocking=True)
                mask_lab = mask_lab.cuda(non_blocking=True).bool()
                images = images.cuda(non_blocking=True)

                if args.mixup_data and epoch < args.max_epochs:
                    images, targets_a, targets_b, lam, index = mixup_data(
                        images, labels, 0.5)
                                                       
                with amp_cm:
                    outputs = model([images], mask_lab, gumbel_scale=gumbel_scalar)
                    with torch.no_grad():
                        old_outputs = old_model([images], mask_lab=None, gumbel_scale=gumbel_scalar)
                        old_logits = (old_outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1))
                        old_logits = old_logits.detach()
                        distances_old = old_outputs["distances"].squeeze()
                    
                    outputs["logits_lab"] = (outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1))
                    logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)

                    if args.similarity_reg:
                        distances = outputs["distances"].squeeze()
                        distances_old = model.head_lab.distance_2_similarity(distances_old)
                        distances = model.head_lab.distance_2_similarity(distances)

                        if args.similarity_reg_on_seen_only:
                            distances_old = distances_old[mask_lab]
                            distances = distances[mask_lab]

                        with torch.no_grad():
                            q = torch.quantile(distances_old.reshape([distances[:,:args.num_prototypes].shape[0], distances[:,:args.num_prototypes].shape[1], -1]), args.perc / 100, dim=2)
                            mask = distances_old >= q[:, :, None, None]
                        
                        sim_reg = ((distances_old - distances[:,:args.num_prototypes]) * mask).view(distances[:,:args.num_prototypes].shape[0], - 1).norm(2, dim=1).sum() / distances.shape[0]
                    else:
                        sim_reg = torch.Tensor([0]).cuda()

                    targets_lab = F.one_hot(
                        labels[mask_lab],
                        num_classes=args.num_labeled_classes).float()

                    targets = torch.zeros_like(logits)

                    # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
                    for v in range(args.num_large_crops):
                        for h in range(args.num_heads):
                            targets[v, h,
                                    mask_lab, :nlc] = targets_lab.type_as(targets)

                            targets[v, h, ~mask_lab,
                                    nlc:] = sk(outputs["logits_unlab"][
                                        v, h, ~mask_lab]).type_as(targets)

                    if args.mixup_data and epoch < args.max_epochs:
                        entropy_loss = lam * cross_entropy_loss(logits, targets, temperature=args.temperature) + \
                            (1 - lam) * cross_entropy_loss(logits, targets[:, :, index], temperature=args.temperature)
                    else:
                        entropy_loss = cross_entropy_loss(logits, targets, temperature=args.temperature)

                    entropy_loss = entropy_loss.mean()

                    if args.use_l2_on_prototypes:
                        proto_l2_loss = torch.norm(model.encoder.prototype_vectors - old_model.encoder.prototype_vectors, p=2)
                    else:
                        proto_l2_loss = torch.Tensor([0]).cuda()
                    
                    if args.not_use_kd_loss:
                        kd_loss = torch.Tensor([0]).cuda()
                    else:
                        kd_loss = KD(args, old_logits[:args.num_large_crops], logits[:args.num_large_crops], mask_lab, T=args.kd_temperature)
                        kd_loss = kd_loss.mean()

                    # TODO: double check if this is correct:
                    orthogonal_loss_head_lab = torch.Tensor([0]).cuda()
                    if args.pp_ortho:
                        for c in range(0, model.head_lab.proto_presence.shape[0], 1000):
                            orthogonal_loss_p = \
                                torch.nn.functional.cosine_similarity(model.head_lab.proto_presence.unsqueeze(2)[c:c+1000],
                                                                        model.head_lab.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                            orthogonal_loss_head_lab += orthogonal_loss_p
                        orthogonal_loss_head_lab = orthogonal_loss_head_lab / (args.num_descriptive * args.num_classes) - 1

                    orthogonal_loss_head_unlab = torch.Tensor([0]).cuda()
                    if args.pp_ortho:
                        for h in range(args.num_heads):
                            for c in range(0, model.head_unlab.prototypes[h].proto_presence.shape[0], 1000):
                                orthogonal_loss_p = \
                                    torch.nn.functional.cosine_similarity(model.head_unlab.prototypes[h].proto_presence.unsqueeze(2)[c:c+1000],
                                                                            model.head_unlab.prototypes[h].proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                                orthogonal_loss_head_unlab += orthogonal_loss_p
                    orthogonal_loss_head_unlab = (orthogonal_loss_head_unlab / (args.num_descriptive * args.num_classes) - 1) / args.num_heads
                    
                    orthogonal_loss = orthogonal_loss_head_lab + orthogonal_loss_head_unlab

                    # jest ok, to samo: labels[mask_lab] == targets[0, 0, mask_lab, :nlc].argmax(axis=-1)
                    proto_presence_lab = outputs['proto_presence_lab'][0][targets[0, 0, mask_lab, :nlc].argmax(axis=-1)]
                    inverted_proto_presence_lab = 1 - proto_presence_lab
                    clst_loss_val_lab = dist_loss(model, outputs['min_distances_lab'][0][mask_lab], proto_presence_lab, args.num_descriptive) 
                    sep_loss_val_lab  = dist_loss(model, outputs['min_distances_lab'][0][mask_lab], inverted_proto_presence_lab, args.num_prototypes - args.num_descriptive) 

                    clst_loss_val_unlab, sep_loss_val_unlab = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
                    for h in range(args.num_heads):
                        proto_presence_unlab = outputs['proto_presence_unlab'][0, h, targets[0, h, ~mask_lab, nlc:].argmax(axis=-1)]
                        inverted_proto_presence_unlab = 1 - proto_presence_unlab
                        clst_loss_val_unlab += dist_loss(model, outputs['min_distances_unlab'][0, h, ~mask_lab], proto_presence_unlab, args.num_descriptive) 
                        sep_loss_val_unlab  += dist_loss(model, outputs['min_distances_unlab'][0, h, ~mask_lab], inverted_proto_presence_unlab, args.num_prototypes - args.num_descriptive)
                    clst_loss_val_unlab /= args.num_heads
                    sep_loss_val_unlab /= args.num_heads

                    clst_loss_val = clst_loss_val_lab + clst_loss_val_unlab
                    sep_loss_val = sep_loss_val_lab + sep_loss_val_unlab
                
                    if args.l1_in_discover_training:
                        l1_mask_lab = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
                        l1_lab = (model.head_lab.last_layer.weight * l1_mask_lab).norm(p=1)

                        l1_unlab = torch.Tensor([0]).cuda()
                        for h in range(args.num_heads):
                            l1_mask_unlab = 1 - torch.t(model.head_unlab.prototypes[h].prototype_class_identity).cuda()
                            l1_unlab += (model.head_unlab.prototypes[h].last_layer.weight * l1_mask_unlab).norm(p=1)
                        l1_unlab = l1_unlab / args.num_heads
                        
                        l1 = l1_lab + l1_unlab
                    else:
                        l1 = torch.Tensor([0]).cuda()

                    loss = entropy_loss*args.entropy_loss_weight + args.alpha * kd_loss + orthogonal_loss + clst_loss_val * clst_weight \
                            + sep_loss_val * sep_weight + 1e-4 * l1 + proto_l2_loss*args.proto_l2_loss_weight + args.lamb * sim_reg
                    
                    if args.amp:
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        # gradient_norms = log_gradient_norms(model)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        
                        if args.freeze_backbone_except_num_extra_prototypes or args.freeze_pretrained_prototypes_except_num_extra_prototypes:
                            model.encoder.prototype_vectors.grad[:args.num_prototypes] = 0
                        
                        gradient_norms = log_gradient_norms(model)
                        optimizer.step()

                    bar.set_postfix(
                        {"loss": "{:.2f}".format(loss.detach().cpu().numpy()[0])})
                    results = {
                        "loss": loss.clone(),
                        "entropy_loss": entropy_loss.clone(),
                        # "cross_entropy_loss_lab": cross_entropy_loss_lab.clone(),
                        # "cross_entropy_loss_unlab": cross_entropy_loss_unlab.clone(),
                        # "cross_entropy_loss_head_lab": cross_entropy_loss_head_lab.clone(),
                        # "cross_entropy_loss_head_unlab": cross_entropy_loss_head_unlab.clone(),
                        "kd_loss": kd_loss.clone(),
                        "orthogonal_loss": orthogonal_loss.clone(),
                        "clst_loss": clst_loss_val.clone(),
                        "sep_loss": sep_loss_val.clone(),
                        "l1": l1.clone(),
                        "proto_l2_loss": proto_l2_loss.clone(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "similarity_reg": sim_reg.clone()
                        # "gradient_norms": gradient_norms
                    }
                    wandb.log(results)

                if args.debugging:
                    break

            # TODO: keeping lr constant is not the best idea:
            # if steps:
            #     lr_scheduler.step()

            best_head = torch.argmin(loss_per_head)
            
            test_results = test(args, model, test_loader, best_head, prefix="test", gumbel_scalar=gumbel_scalar)
            
            train_results = test(args,
                                model,
                                train_val_loader,
                                best_head,
                                prefix="train",
                                gumbel_scalar=gumbel_scalar)
            
            train_dataloader_results = test(args,
                                            model,
                                            train_loader,
                                            best_head,
                                            prefix="train_loader",
                                            gumbel_scalar=gumbel_scalar)

            wandb.log(train_results)
            wandb.log(test_results)
            wandb.log(train_dataloader_results)


            if epoch >= args.max_epochs:
                args.save_model_every = 1
            
            if args.save_model_every and epoch % args.save_model_every == 0:
                model_to_save = model.module if hasattr(model, "module") else model
                state_save = {
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(
                    state_save,
                    os.path.join(args.model_save_dir, f"checkpoint_epoch_{epoch}.pth"))

            # save model
            if args.save_model:
                model_to_save = model.module if hasattr(model, "module") else model
                state_save = {
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'is_push': False
                }
                torch.save(
                    state_save,
                    os.path.join(args.model_save_dir, "latest_checkpoint.pth"))
                
                # TODO: do I really want to check accuracy on train data? We always look at test data!
                if train_results["train/novel/avg"] > best_scores["acc"]:
                    best_scores["acc"] = train_results["train/novel/avg"]
                    best_scores.update(train_results)
                    torch.save(
                        state_save,
                        os.path.join(args.model_save_dir, "best_checkpoint.pth"))

            # log
            lr = optimizer.param_groups[0]["lr"]
            print("--Comment-{}--Epoch-[{}/{}]--LR-[{}]--Train-Novel-[{:.2f}]--"
                "Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".format(
                    f"{args.comment}_{args.time_run}", epoch, args.max_epochs, lr,
                    train_results["train/novel/avg"] * 100,
                    test_results["test/all/avg"] * 100,
                    test_results["test/novel/avg"] * 100,
                    test_results["test/seen/avg"] * 100))
        
            if args.debugging and not args.debugging_extra_epochs:
                break
            
        ####################################
        #       push (head_lab) step       #
        #       (leave original names)     #
        ####################################
    print('Model push')

    if args.only_push and args.only_push_without_finetuning:
        push_assuming_knowledge_of_all_labels(args, model, train_push_loader, train_val_loader, test_loader)
        print('Finished only push without finetuning')
        sys.exit()
        
    if args.only_push and not args.restart_from_checkpoint:
        # load best model with path:
        print('Loading best model')
        # os.path.join(args.only_push_model_dir, 'best_checkpoint.pth')
        state = torch.load(args.only_push_model_path)
        model.load_state_dict(state['state_dict'])
        model = model.cuda()

    # CHECK ACCURACY BEFORE PUSH and get mapped labels:
    # test_results = test(args, model, test_loader, best_head=0, prefix="test", gumbel_scalar=10e3)
    
    # train_results = test(args,
    #                     model,
    #                     train_val_loader,
    #                     best_head=0,
    #                     prefix="train",
    #                     gumbel_scalar=10e3)
    
    # print("--Accuracy before push-Train-Novel-[{:.2f}]--"
    # "Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".format(
    #     train_results["train/novel/avg"] * 100,
    #     test_results["test/all/avg"] * 100,
    #     test_results["test/novel/avg"] * 100,
    #     test_results["test/seen/avg"] * 100))

    map_labels = get_map_labels(args, model, train_push_loader)

    proto_img_dir = f'{args.model_save_dir}/{args.proto_img_dir}/'
    Path(proto_img_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    global_min_proto_dist = np.full(model.encoder.num_prototypes + model.encoder.num_extra_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [model.head_lab.num_prototypes + model.encoder.num_extra_prototypes,
            model.encoder.prototype_shape[1],
            model.encoder.prototype_shape[2],
            model.encoder.prototype_shape[3]])

    proto_rf_boxes = np.full(shape=[model.head_lab.num_prototypes + model.encoder.num_extra_prototypes, 6],
                                fill_value=-1)
    proto_bound_boxes = np.full(shape=[model.head_lab.num_prototypes + model.encoder.num_extra_prototypes, 6],
                                        fill_value=-1)

    start_index_of_search_batch = 0
    with torch.no_grad():
        for push_iter, (images, labels, _, mask_lab) in enumerate(train_push_loader):

            images = images.cuda()
            labels = labels.cuda()
            mask_lab = mask_lab[:, 0].bool().cuda()
            
            search_batch_input = images[mask_lab]
            search_y = labels[mask_lab]
            mask_lab = mask_lab[mask_lab]

            if search_batch_input.shape[0] == 0:
                continue

            update_prototypes_on_batch(search_batch_input=search_batch_input, 
                                        start_index_of_search_batch=start_index_of_search_batch,
                                        model=model,
                                        global_min_proto_dist=global_min_proto_dist,
                                        global_min_fmap_patches=global_min_fmap_patches,
                                        proto_rf_boxes=proto_rf_boxes,
                                        proto_bound_boxes=proto_bound_boxes,
                                        class_specific=True,
                                        search_y=search_y,
                                        prototype_layer_stride=1,
                                        dir_for_saving_prototypes=proto_img_dir,
                                        prototype_img_filename_prefix='prototype-img',
                                        prototype_self_act_filename_prefix='prototype-self-act',
                                        prototype_activation_function_in_numpy=None,
                                        h=None,
                                        mask_lab=mask_lab[mask_lab])
            
            start_index_of_search_batch += mask_lab.sum()


    # jeszcze teraz nie nadpisuję prototypów
    # a terach chciałbym zebrać dodatkowe prototypy na podstawie nieznanych label-i
    # Zostaną one dodane jedynie jeśli będą miały miejszy dystans 

    start_index_of_search_batch = 0
    collect_to_verify_search_y, collect_to_verify_labels = [], []

    with torch.no_grad():
        for push_iter, (images, labels, _, mask_lab) in enumerate(train_push_loader):

            if (~mask_lab).sum().item() == 0:
                continue

            images = images.cuda()
            labels = labels.cuda()
            mask_lab = mask_lab[:, 0].bool().cuda()
            
            search_batch_input = images[~mask_lab].cuda()

            if search_batch_input.shape[0] == 0:
                continue

            outputs = model(search_batch_input, mask_lab=mask_lab[~mask_lab], gumbel_scale=10e3)
            only_logits_unlab = outputs['logits_unlab']
            only_unlab_preds = only_logits_unlab.max(dim=-1)[1]

            # note: head are 'checked' one after the other
            for h in range(args.num_heads):

                only_unlab_preds_h = only_unlab_preds[h]

                search_y = np.array([map_labels[h][pred.item()] for pred in only_unlab_preds_h])

                # collect_to_verify_search_y.extend((search_y + 100).tolist())
                # collect_to_verify_labels.extend(labels[~mask_lab].tolist())

                update_prototypes_on_batch(search_batch_input=search_batch_input, 
                                            start_index_of_search_batch=start_index_of_search_batch,
                                            model=model,
                                            global_min_proto_dist=global_min_proto_dist,
                                            global_min_fmap_patches=global_min_fmap_patches,
                                            proto_rf_boxes=proto_rf_boxes,
                                            proto_bound_boxes=proto_bound_boxes,
                                            class_specific=True,
                                            search_y=search_y,
                                            prototype_layer_stride=1,
                                            dir_for_saving_prototypes=proto_img_dir,
                                            prototype_img_filename_prefix='prototype-img',
                                            prototype_self_act_filename_prefix='prototype-self-act',
                                            prototype_activation_function_in_numpy=None,
                                            h=h,
                                            mask_lab=mask_lab[~mask_lab])


            start_index_of_search_batch += (~mask_lab).sum()

            if args.debugging:
                break

    # dopiero teraz nadpisuję prototypy
    prototype_update = np.reshape(global_min_fmap_patches,
                                    tuple(model.encoder.prototype_shape))
    model.encoder.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())


    print('Fine-tuning')
    
    best_push_scores = {"epoch": 0, "acc": 0}
    for tune_epoch in range(start_epoch, args.max_tune_epoch):
        model.train()

        bar = tqdm(train_loader)

        for batch in bar:
            images, labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            label_p = labels.numpy().tolist()

            labels = labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            images = images.cuda(non_blocking=True)

            if args.mixup_data:
                images, targets_a, targets_b, lam, index = mixup_data(
                    images, labels, 0.5)

            if args.normalize_last_layers:
                model.normalize_prototypes()    

            outputs = model([images], mask_lab=mask_lab, gumbel_scale=10e3)
            with torch.no_grad():
                old_outputs = old_model([images], mask_lab=None, gumbel_scale=10e3)
                old_logits = (old_outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1))
                old_logits = old_logits.detach()
                distances_old = old_outputs["distances"].squeeze()
 
            # gather outputs
            outputs["logits_lab"] = (outputs["logits_lab"].unsqueeze(1).expand(-1, args.num_heads, -1, -1))
            logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
            
            if args.similarity_reg:
                distances = outputs["distances"].squeeze()

                distances_old = model.head_lab.distance_2_similarity(distances_old)
                distances = model.head_lab.distance_2_similarity(distances)

                if args.similarity_reg_on_seen_only:
                    distances_old = distances_old[mask_lab]
                    distances = distances[mask_lab]

                with torch.no_grad():
                    q = torch.quantile(distances_old.reshape([distances[:,:args.num_prototypes].shape[0], distances[:,:args.num_prototypes].shape[1], -1]), args.perc / 100, dim=2)
                    mask = distances_old >= q[:, :, None, None]
                
                sim_reg = ((distances_old - distances[:,:args.num_prototypes]) * mask).view(distances[:,:args.num_prototypes].shape[0], - 1).norm(2, dim=1).sum() / distances.shape[0]
            else:
                sim_reg = torch.Tensor([0]).cuda()

            # create targets
            targets_lab = F.one_hot(
                labels[mask_lab],
                num_classes=args.num_labeled_classes).float()

            targets = torch.zeros_like(logits)

            # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
            for v in range(args.num_large_crops):
                for h in range(args.num_heads):
                    targets[v, h,
                            mask_lab, :nlc] = targets_lab.type_as(targets)

                    targets[v, h, ~mask_lab,
                            nlc:] = sk(outputs["logits_unlab"][
                                v, h, ~mask_lab]).type_as(targets)

            if args.mixup_data:
                entropy_loss = lam * cross_entropy_loss(logits, targets, temperature=args.temperature) + \
                    (1 - lam) * cross_entropy_loss(logits, targets[:, :, index], temperature=args.temperature)
            else:
                entropy_loss = cross_entropy_loss(logits, targets, temperature=args.temperature)

            entropy_loss = entropy_loss.mean()
            cross_entropy_loss_lab = cross_entropy_loss(logits[:, :, mask_lab, :nlc], targets[:, :, mask_lab, :nlc], temperature=args.temperature).mean()
            cross_entropy_loss_unlab = cross_entropy_loss(logits[:, :, ~mask_lab, nlc:], targets[:, :, ~mask_lab, nlc:], temperature=args.temperature).mean()

            cross_entropy_loss_head_lab = cross_entropy_loss(outputs["logits_lab"], targets[:, :, :, :nlc], temperature=args.temperature).mean()
            cross_entropy_loss_head_unlab = cross_entropy_loss(outputs["logits_unlab"], targets[:, :, :, nlc:], temperature=args.temperature).mean()
            
            if args.not_use_kd_loss:
                kd_loss = torch.Tensor([0]).cuda()
            else:
                kd_loss = KD(args, old_logits[:args.num_large_crops], logits[:args.num_large_crops], mask_lab, T=args.kd_temperature)
                kd_loss = kd_loss.mean()

            # TODO: double check if this is correct:
            orthogonal_loss_head_lab = torch.Tensor([0]).cuda()
            if args.pp_ortho:
                for c in range(0, model.head_lab.proto_presence.shape[0], 1000):
                    orthogonal_loss_p = \
                        torch.nn.functional.cosine_similarity(model.head_lab.proto_presence.unsqueeze(2)[c:c+1000],
                                                                model.head_lab.proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                    orthogonal_loss_head_lab += orthogonal_loss_p
                orthogonal_loss_head_lab = orthogonal_loss_head_lab / (args.num_descriptive * args.num_classes) - 1

            orthogonal_loss_head_unlab = torch.Tensor([0]).cuda()
            if args.pp_ortho:
                for h in range(args.num_heads):
                    for c in range(0, model.head_unlab.prototypes[h].proto_presence.shape[0], 1000):
                        orthogonal_loss_p = \
                            torch.nn.functional.cosine_similarity(model.head_unlab.prototypes[h].proto_presence.unsqueeze(2)[c:c+1000],
                                                                    model.head_unlab.prototypes[h].proto_presence.unsqueeze(-1)[c:c+1000], dim=1).sum()
                        orthogonal_loss_head_unlab += orthogonal_loss_p
            orthogonal_loss_head_unlab = (orthogonal_loss_head_unlab / (args.num_descriptive * args.num_classes) - 1) / args.num_heads
            
            orthogonal_loss = orthogonal_loss_head_lab + orthogonal_loss_head_unlab

            # jest ok, to samo: labels[mask_lab] == targets[0, 0, mask_lab, :nlc].argmax(axis=-1)
            proto_presence_lab = outputs['proto_presence_lab'][0][targets[0, 0, mask_lab, :nlc].argmax(axis=-1)]
            inverted_proto_presence_lab = 1 - proto_presence_lab
            clst_loss_val_lab = dist_loss(model, outputs['min_distances_lab'][0][mask_lab], proto_presence_lab, args.num_descriptive) 
            sep_loss_val_lab  = dist_loss(model, outputs['min_distances_lab'][0][mask_lab], inverted_proto_presence_lab, args.num_prototypes - args.num_descriptive) 

            clst_loss_val_unlab, sep_loss_val_unlab = torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()
            for h in range(args.num_heads):
                proto_presence_unlab = outputs['proto_presence_unlab'][0, h, targets[0, h, ~mask_lab, nlc:].argmax(axis=-1)]
                inverted_proto_presence_unlab = 1 - proto_presence_unlab
                clst_loss_val_unlab += dist_loss(model, outputs['min_distances_unlab'][0, h, ~mask_lab], proto_presence_unlab, args.num_descriptive) 
                sep_loss_val_unlab  += dist_loss(model, outputs['min_distances_unlab'][0, h, ~mask_lab], inverted_proto_presence_unlab, args.num_prototypes - args.num_descriptive)
            clst_loss_val_unlab /= args.num_heads
            sep_loss_val_unlab /= args.num_heads

            clst_loss_val = clst_loss_val_lab + clst_loss_val_unlab
            sep_loss_val = sep_loss_val_lab + sep_loss_val_unlab
        
            l1_mask_lab = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
            l1_lab = (model.head_lab.last_layer.weight * l1_mask_lab).norm(p=1)

            l1_unlab = torch.Tensor([0]).cuda()
            for h in range(args.num_heads):
                l1_mask_unlab = 1 - torch.t(model.head_unlab.prototypes[h].prototype_class_identity).cuda()
                l1_unlab += (model.head_unlab.prototypes[h].last_layer.weight * l1_mask_unlab).norm(p=1)
            l1_unlab = l1_unlab / args.num_heads
            
            l1 = l1_lab + l1_unlab

            loss = entropy_loss*args.entropy_loss_weight + args.alpha * kd_loss + orthogonal_loss + clst_loss_val * clst_weight + sep_loss_val * sep_weight + 1e-4 * l1 + args.lamb * sim_reg
                    
            # ===================backward====================
            push_optimizer.zero_grad()
            loss.backward()

            if args.freeze_backbone_except_num_extra_prototypes or args.freeze_pretrained_prototypes_except_num_extra_prototypes:
                model.encoder.prototype_vectors.grad[:args.num_prototypes] = 0

            push_optimizer.step()
            
            bar.set_postfix(
                {"loss": "{:.2f}".format(loss.item())})
            
            results = {
                "loss": loss.clone(),
                "entropy_loss": entropy_loss.clone(),
                "cross_entropy_loss_lab": cross_entropy_loss_lab.clone(),
                "cross_entropy_loss_unlab": cross_entropy_loss_unlab.clone(),
                "cross_entropy_loss_head_lab": cross_entropy_loss_head_lab.clone(),
                "cross_entropy_loss_head_unlab": cross_entropy_loss_head_unlab.clone(),
                "kd_loss": kd_loss.clone(),
                "orthogonal_loss": orthogonal_loss.clone(),
                "clst_loss": clst_loss_val.clone(),
                "sep_loss": sep_loss_val.clone(),
                "l1": l1.clone(),
                "lr": push_optimizer.param_groups[0]["lr"],
                "similarity_reg": sim_reg.clone()
                # "gradient_norms": gradient_norms
            }
            wandb.log(results)

            if args.debugging:
                break

        ####################################
        #          validation step         #
        ####################################


        # # TODO: best head: get loss_per_head
        best_head = torch.argmin(loss_per_head)
        
        test_results, tst_loss = test(args, model, test_loader, best_head, prefix="test",
                                    gumbel_scalar=10e3, return_extra=True)
        train_results = test(args,
                            model,
                            train_val_loader,
                            best_head,
                            prefix="train",
                            gumbel_scalar=10e3)
        
        train_dataloader_results = test(args,
                                        model,
                                        train_loader,
                                        best_head,
                                        prefix="train_loader",
                                        gumbel_scalar=10e3)

        wandb.log(train_results)
        wandb.log(test_results)
        wandb.log(train_dataloader_results)

        if args.save_model_every and tune_epoch % args.save_model_every == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            state_save = {
                'tune_epoch': tune_epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'push_optimizer': push_optimizer.state_dict(),
            }
            torch.save(
                state_save,
                os.path.join(args.model_save_dir, f"push_checkpoint_tune_epoch_{tune_epoch}.pth"))

        # save model
        if args.save_model:
            model_to_save = model.module if hasattr(model, "module") else model
            state_save = {
                'epoch': tune_epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'optimizer': push_optimizer.state_dict(),
                'is_push': True
            }
            torch.save(
                state_save,
                os.path.join(args.model_save_dir, "push_latest_checkpoint.pth"))
            if train_results["train/novel/avg"] > best_push_scores["acc"]:
                best_push_scores["acc"] = train_results["train/novel/avg"]
                best_push_scores.update(train_results)
                torch.save(
                    state_save,
                    os.path.join(args.model_save_dir, "push_best_checkpoint.pth"))
        
        lr = push_optimizer.param_groups[0]["lr"]
        print("--Comment-{}--Tune_epoch-[{}/{}]--LR-[{}]--Train-Novel-[{:.2f}]--"
            "Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".format(
                f"{args.comment}_{args.time_run}", tune_epoch, args.max_epochs, lr,
                train_results["train/novel/avg"] * 100,
                test_results["test/all/avg"] * 100,
                test_results["test/novel/avg"] * 100,
                test_results["test/seen/avg"] * 100))
        ####################################
        #  scheduler and early stop step   #
        ####################################
        if (tune_epoch + 1) % 5 == 0:
            adjust_learning_rate(push_optimizer, 0.95)

        if args.debugging:
            break

    wandb.finish()
    print('Finished training model. Have a nice day :)')


def get_map_labels(args, model, train_push_loader, yt_map=False):
    '''
    Returns mapping from pred labels to most optimal pred labels
    for each head.

    if yt_map is True, returns mapping for true labels to 
    most optimal true labels
    '''

    model.eval()

    all_only_unlab_labels = None
    all_only_unlab_preds = None
    with torch.no_grad():
        
        for images, labels, _, mask_lab in train_push_loader:

            images = images.cuda()
            labels = labels.cuda()
            mask_lab = mask_lab[:, 0].bool().cuda()

            if images[~mask_lab].size(0) == 0:
                continue

            outputs = model(images[~mask_lab], mask_lab[~mask_lab], gumbel_scale=10e3)
            
            only_logits_unlab = outputs['logits_unlab']
            only_unlab_preds = only_logits_unlab.max(dim=-1)[1]

            only_unlab_labels = labels[~mask_lab].unsqueeze(0).expand(args.num_heads, -1)

            if all_only_unlab_labels is None:
                all_only_unlab_labels = only_unlab_labels
                all_only_unlab_preds = only_unlab_preds
            else:
                all_only_unlab_labels = torch.cat([all_only_unlab_labels, only_unlab_labels], dim=-1)
                all_only_unlab_preds = torch.cat([all_only_unlab_preds, only_unlab_preds], dim=-1)
            

    all_only_unlab_labels = all_only_unlab_labels.detach().cpu().numpy()
    all_only_unlab_preds = all_only_unlab_preds.detach().cpu().numpy()
    
    # for validation:
    y_pred_d = {h:None for h in range(args.num_heads)}

    map_labels = {h:None for h in range(args.num_heads)}
    true_labels_map = {h:None for h in range(args.num_heads)}
    for h in range(args.num_heads):

        y_true = all_only_unlab_labels[h].astype(int) - 100
        y_pred = all_only_unlab_preds[h].astype(int)

        # for validation:
        y_pred_d[h] = y_pred.copy()

        assert y_pred.size == y_true.size
        D = max(y_true.max()-min(y_true), y_true.max()-min(y_true)) + 1
        w = np.zeros((D, D), dtype=int)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        ind = linear_sum_assignment(w.max() - w)
        ind = np.vstack(ind).T

        # it is mapping from pred to 
        # most optimal pred.
        map_labels[h] = {i: j for i, j in ind}

        true_labels_map[h] = {j: i for i, j in ind}

    if not yt_map:
        return map_labels
    else:
        return true_labels_map
    

def update_prototypes_on_batch(search_batch_input, start_index_of_search_batch,
                               model,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               class_specific=True,
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               h=None,
                               top_patches_data=None,
                               n_top_patches=None,
                               mask_lab=None,
                               map_labels=None,
                               assuming_knowledge_of_all_labels=False):
    model.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()

        # to jest po prostu dystans między prototypami a obrazami w batch-u:
        # (batch_size, num_prototypes, wysokość latentu, szerokość latentu)
        proto_dist_torch = model.encoder.prototype_distances(search_batch, mask_lab=mask_lab)
        # to jest po prostu latent: (batch_size, długość latentu, wysokość latentu, szerokość latentu)
        protoL_input_torch = model.encoder.conv_features(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model.encoder.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    # nie wiem co to za maksymalny dystans??? I niby względem czego jest on maksymalny?
    # wygląda jakby był jego maksimum wyznaczał promienień sfery: 256 (długości latentu)
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if class_specific:
        if h is None and not assuming_knowledge_of_all_labels:
            # to są numery prototypów przypisanych do danej klasy:
            map_class_to_prototypes = model.head_lab.get_map_class_to_prototypes()
        elif assuming_knowledge_of_all_labels:
            map_class_to_prototypes_head_lab = model.head_lab.get_map_class_to_prototypes()
            map_class_to_prototypes_head_unlab = model.head_unlab.prototypes[0].get_map_class_to_prototypes() #TODO: CHOOSING HEAD 0 FOR NOW GOOD REASON (BETTER CHOOSE BEST HEAD)
            # swapping the ortder in accordance to the most optimal pred assignment:
            map_class_to_prototypes_head_unlab = model.head_unlab.prototypes[0].get_map_class_to_prototypes()[list(map_labels[0].values()),:]
            map_class_to_prototypes = np.concatenate([map_class_to_prototypes_head_lab, map_class_to_prototypes_head_unlab], axis=0)

        else:
            map_class_to_prototypes = model.head_unlab.prototypes[h].get_map_class_to_prototypes()
            
        # to jest słownik, gdzie kluczem jest numer prototypu, a wartością pusta 
        # lista na indeksy obrazów, które są przypisane do tego prototypu
        # indexy te wskazują na położenie obrazu w search_batch
        protype_to_img_index_dict = {key: [] for key in range(n_prototypes)}
        
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            for prototype in map_class_to_prototypes[img_label]: #img_label TODO: remember to change it to img_label
                protype_to_img_index_dict[prototype].append(img_index)

            # bardziej czytelnie przepisane powyżej
            # [protype_to_img_index_dict[prototype].append(
            #     img_index) for prototype in map_class_to_prototypes[img_label]]

    for j in range(n_prototypes):
        if class_specific:
            
            # jeżeli nie ma żadnego obrazu przypisanego
            # w tym batch-u do prototypu, to kontynuuj
            if len(protype_to_img_index_dict[j]) == 0:
                continue

            # jeżeli jest class_specific, to:
            # bierzemy odległości między prototypem j, a obrazami z batch-a.
            # Interesują nas w tej chwili odległości między prototypem j, a tymi obrazami.
            # Czyli najpier bierzemy indeksy obrazów: [protype_to_img_index_dict[j],
            # a następnie wybieramy prototyp j: [:, j].
            # więc protot_dist_j.shape np.: (30, 7, 7) *pamiętaj, że train_push_loader 
            # jest shuffle=False i każda klasa ma 30 obrazów.
            proto_dist_j = proto_dist_[protype_to_img_index_dict[j]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        # Teraz chemy sprawdzić czy dla danego prototypu j
        # jego odległość od przypisanych mu obrazów jest mniejsza
        # niż odległości od obrazów z poprzednich batch-ów.
        # W tym celu szukamy najmniejszej wartości w proto_dist_j
        batch_min_proto_dist_j = np.amin(proto_dist_j)

        # Jeśli ta wartość jest niejsza niż poprzednia 
        if batch_min_proto_dist_j < global_min_proto_dist[j]:

            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''

                batch_argmin_proto_dist_j[0] = protype_to_img_index_dict[j][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * \
                prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * \
                prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            # uzyskujemy wektor (głębokość_latentu, 1, 1)
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            # aktualizujemy najmniejszą odległość:
            global_min_proto_dist[j] = batch_min_proto_dist_j
            # aktualizujemy prototyp:
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = model.proto_layer_rf_info

            # Tutaj chielibyśmy się dowiedzieć jakie wymiary ma patch,
            # który generuje reprezentację prototypu j.
            layer_filter_sizes, layer_strides, layer_paddings = model.encoder.features.conv_info()
            protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                           prototype_kernel_size=1)
            
            # receptive field: [278, 0, 224, 0, 214], gdzie:
            # 278 - numer obrazu w batch-u
            # 0 - początek wysokości
            # 224 - koniec wysokości
            # 0 - początek szerokości
            # 214 - koniec szerokości
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.detach().cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # if top_patches_data is not None and n_top_patches is not None:
            #     update_top_patches(top_patches_data[j], batch_min_proto_dist_j, batch_min_fmap_patch_j, rf_img_j, n_top_patches)

            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]

            # TODO: chnage head_lab to head_unlab if necessary:
            if model.head_lab.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model.head_lab.epsilon))
            elif model.head_lab.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)


def update_prototypes_on_batch_test(search_batch_input, start_index_of_search_batch,
                               model,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               class_specific=True,
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               map_labels=None):
    
    model.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        proto_dist_torch = model.encoder.prototype_distances(search_batch, mask_lab=None)
        protoL_input_torch = model.encoder.conv_features(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model.encoder.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if class_specific:
        # map_class_to_prototypes = model.get_map_class_to_prototypes()

        map_class_to_prototypes_head_lab = model.head_lab.get_map_class_to_prototypes()
        map_class_to_prototypes_head_unlab = model.head_unlab.prototypes[0].get_map_class_to_prototypes() #TODO: CHOOSING HEAD 0 FOR NOW GOOD REASON (BETTER CHOOSE BEST HEAD)
        # swapping the ortder in accordance to the most optimal pred assignment:
        map_class_to_prototypes_head_unlab = model.head_unlab.prototypes[0].get_map_class_to_prototypes()[list(map_labels[0].values()),:]
        map_class_to_prototypes = np.concatenate([map_class_to_prototypes_head_lab, map_class_to_prototypes_head_unlab], axis=0)
        
        protype_to_img_index_dict = {key: [] for key in range(n_prototypes)}
        # img_y is the image's integer label

        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            for prototype in map_class_to_prototypes[img_label]:
                protype_to_img_index_dict[prototype].append(img_index)
            
            # [protype_to_img_index_dict[prototype].append(
            #     img_index) for prototype in map_class_to_prototypes[img_label]]

    for j in range(n_prototypes):
        if class_specific:
            # target_class is the class of the class_specific prototype

            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(protype_to_img_index_dict[j]) == 0:
                continue
            proto_dist_j = proto_dist_[protype_to_img_index_dict[j]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        batch_min_proto_dist_j = np.amin(proto_dist_j)

        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''

                batch_argmin_proto_dist_j[0] = protype_to_img_index_dict[j][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * \
                prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * \
                prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

           # get the receptive field boundary of the image patch
            # that generates the representation
            # protoL_rf_info = model.proto_layer_rf_info
            layer_filter_sizes, layer_strides, layer_paddings = model.encoder.features.conv_info()
            protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                           prototype_kernel_size=1)
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if model.head_lab.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model.head_lab.epsilon))
            elif model.head_lab.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)


def find_top_n_closest_patches_for_each_prototype(search_batch_input, start_index_of_search_batch,
                               model,
                               global_min_proto_dist,  # this will be updated
                               global_min_fmap_patches,  # this will be updated
                               proto_rf_boxes,  # this will be updated
                               proto_bound_boxes,  # this will be updated
                               class_specific=True,
                               search_y=None,  # required if class_specific == True
                               num_classes=None,  # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None,
                               h=None,
                               prototype_heaps_dict=None,
                               n_top_patches=None,
                               global_unique_set=None,
                               labels=None,
                               mask_lab=None,
                               n_prototypes=None,
                               start_from_prototype=0):

    model.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()

        # to jest po prostu dystans między prototypami a obrazami w batch-u:
        # (batch_size, num_prototypes, wysokość latentu, szerokość latentu)
        proto_dist_torch = model.encoder.prototype_distances(search_batch, mask_lab=mask_lab)
        # to jest po prostu latent: (batch_size, długość latentu, wysokość latentu, szerokość latentu)
        protoL_input_torch = model.encoder.conv_features(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    prototype_shape = model.encoder.prototype_shape
    # n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    # nie wiem co to za maksymalny dystans??? I niby względem czego jest on maksymalny?
    # wygląda jakby był jego maksimum wyznaczał promienień sfery: 256 (długości latentu)
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if class_specific:
        if h is None:
            # to są numery prototypów przypisanych do danej klasy:
            map_class_to_prototypes = model.head_lab.get_map_class_to_prototypes()
        else:
            map_class_to_prototypes = model.head_unlab.prototypes[h].get_map_class_to_prototypes()
            
        # to jest słownik, gdzie kluczem jest numer prototypu, a wartością pusta 
        # lista na indeksy obrazów, które są przypisane do tego prototypu
        # indexy te wskazują na położenie obrazu w search_batch
        protype_to_img_index_dict = {key: [] for key in range(n_prototypes)}
        
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            for prototype in map_class_to_prototypes[img_label]:
                protype_to_img_index_dict[prototype].append(img_index)

            # bardziej czytelnie przepisane powyżej
            # [protype_to_img_index_dict[prototype].append(
            #     img_index) for prototype in map_class_to_prototypes[img_label]]

    for j in range(start_from_prototype, n_prototypes):
        if class_specific:
            
            # jeżeli nie ma żadnego obrazu przypisanego
            # w tym batch-u do prototypu, to kontynuuj
            if len(protype_to_img_index_dict[j]) == 0:
                continue

            # jeżeli jest class_specific, to:
            # bierzemy odległości między prototypem j, a obrazami z batch-a.
            # Interesują nas w tej chwili odległości między prototypem j, a tymi obrazami.
            # Czyli najpier bierzemy indeksy obrazów: [protype_to_img_index_dict[j],
            # a następnie wybieramy prototyp j: [:, j].
            # więc protot_dist_j.shape np.: (30, 7, 7) *pamiętaj, że train_push_loader 
            # jest shuffle=False i każda klasa ma 30 obrazów.
            proto_dist_j = proto_dist_[protype_to_img_index_dict[j]][:, j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        # Teraz chcę znaleźć top n najbliższych patchy dla danego prototypu j
        # w tym batch-u. W tym celu muszę znaleźć n zdjęć, które są najbliżej prototypu j.
        for ith_index, distances_of_prototype_j_and_ith_image_latent in enumerate(proto_dist_j):

            # distances_of_prototype_j_and_ith_image_latent is of shape (7,7) and I need to find the smallest value in it:
            smallest_distance_between_prototype_j_and_ith_image_latent = np.amin(distances_of_prototype_j_and_ith_image_latent)
            # and the index of the smallest value:
            argmin_h_w = np.unravel_index(np.argmin(distances_of_prototype_j_and_ith_image_latent, axis=None), 
                                            distances_of_prototype_j_and_ith_image_latent.shape)


            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''

                img_index_in_the_entire_search_batch = protype_to_img_index_dict[j][ith_index]
                ith_index = img_index_in_the_entire_search_batch

            fmap_height_start_index = argmin_h_w[0] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = argmin_h_w[1] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w


            # uzyskujemy wektor (głębokość_latentu, 1, 1)
            batch_min_fmap_patch_j = protoL_input_[ith_index,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]




            # Tutaj chielibyśmy się dowiedzieć jakie wymiary ma patch,
            # który generuje reprezentację prototypu j.
            layer_filter_sizes, layer_strides, layer_paddings = model.encoder.features.conv_info()
            protoL_rf_info = compute_proto_layer_rf_info_v2(224, layer_filter_sizes, layer_strides, layer_paddings,
                                           prototype_kernel_size=1)
            
            # receptive field: [278, 0, 224, 0, 214], gdzie:
            # 278 - numer obrazu w batch-u
            # 0 - początek wysokości
            # 224 - koniec wysokości
            # 0 - początek szerokości
            # 214 - koniec szerokości

            batch_argmin_proto_dist_j = [ith_index, argmin_h_w[0], argmin_h_w[1]]

            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.detach().cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            original_img_j = (original_img_j - np.min(original_img_j)) / np.max(original_img_j - np.min(original_img_j))
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[ith_index, j, :, :]

            # TODO: chnage head_lab to head_unlab if necessary:
            if model.head_lab.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + model.head_lab.epsilon))
            elif model.head_lab.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # overlay (upsampled) self activation on original image and save the result
            rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
            rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap

            global_img_index = start_index_of_search_batch+ith_index

            unique_key = int(f'{global_img_index}{j}{argmin_h_w[0]}{argmin_h_w[1]}')

            if unique_key not in global_unique_set:

                sy = None

                heapq.heappush(prototype_heaps_dict[j], (-smallest_distance_between_prototype_j_and_ith_image_latent, 
                                    random.random(), h, sy,
                                    labels[ith_index].item(),
                                    #   j, ith_index, 
                                    batch_min_fmap_patch_j, rf_img_j, proto_img_j,
                                    overlayed_original_img_j))

                if len(prototype_heaps_dict[j]) > n_top_patches:
                    heapq.heappop(prototype_heaps_dict[j])

                global_unique_set.add(unique_key)


def push_assuming_knowledge_of_all_labels(args, model, train_push_loader, train_val_loader, test_loader):

    # getting mapping:
    map_labels = get_map_labels(args, model, train_push_loader)

    # load best model with path:
    print('Loading best model')
    # os.path.join(args.only_push_model_dir, 'best_checkpoint.pth')
    state = torch.load(args.only_push_model_path)
    model.load_state_dict(state['state_dict'])
    model = model.cuda()

    # CHECK ACCURACY BEFORE PUSH and get mapped labels:
    # test_results = test(args, model, test_loader, best_head=0, prefix="test", gumbel_scalar=10e3)
    
    # train_results = test(args,
    #                     model,
    #                     train_val_loader,
    #                     best_head=0,
    #                     prefix="train",
    #                     gumbel_scalar=10e3)
    
    # print("--Accuracy before push-Train-Novel-[{:.2f}]--"
    # "Test-All-[{:.2f}]--Novel-[{:.2f}]--Seen-[{:.2f}]".format(
    #     train_results["train/novel/avg"] * 100,
    #     test_results["test/all/avg"] * 100,
    #     test_results["test/novel/avg"] * 100,
    #     test_results["test/seen/avg"] * 100))

    proto_img_dir = f'{args.model_save_dir}/{args.proto_img_dir}/'
    Path(proto_img_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    global_min_proto_dist = np.full(model.encoder.num_prototypes + model.encoder.num_extra_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [model.head_lab.num_prototypes + model.encoder.num_extra_prototypes,
            model.encoder.prototype_shape[1],
            model.encoder.prototype_shape[2],
            model.encoder.prototype_shape[3]])

    proto_rf_boxes = np.full(shape=[model.head_lab.num_prototypes + model.encoder.num_extra_prototypes, 6],
                                fill_value=-1)
    proto_bound_boxes = np.full(shape=[model.head_lab.num_prototypes + model.encoder.num_extra_prototypes, 6],
                                        fill_value=-1)

    start_index_of_search_batch = 0
    with torch.no_grad():
        for push_iter, (images, labels, _, mask_lab) in enumerate(train_push_loader):

            images = images.cuda()
            labels = labels.cuda()
            mask_lab = mask_lab[:, 0].bool().cuda()
            
            search_batch_input = images
            search_y = labels
            mask_lab = mask_lab

            # update_prototypes_on_batch(search_batch_input=search_batch_input, 
            #                             start_index_of_search_batch=start_index_of_search_batch,
            #                             model=model,
            #                             global_min_proto_dist=global_min_proto_dist,
            #                             global_min_fmap_patches=global_min_fmap_patches,
            #                             proto_rf_boxes=proto_rf_boxes,
            #                             proto_bound_boxes=proto_bound_boxes,
            #                             class_specific=True,
            #                             search_y=search_y,
            #                             prototype_layer_stride=1,
            #                             dir_for_saving_prototypes=proto_img_dir,
            #                             prototype_img_filename_prefix='prototype-img',
            #                             prototype_self_act_filename_prefix='prototype-self-act',
            #                             prototype_activation_function_in_numpy=None,
            #                             h=None,
            #                             mask_lab=None,
            #                             map_labels=map_labels,
            #                             assuming_knowledge_of_all_labels=True)
            
            update_prototypes_on_batch_test(search_batch_input=search_batch_input, 
                                        start_index_of_search_batch=start_index_of_search_batch,
                                        model=model,
                                        global_min_proto_dist=global_min_proto_dist,
                                        global_min_fmap_patches=global_min_fmap_patches,
                                        proto_rf_boxes=proto_rf_boxes,
                                        proto_bound_boxes=proto_bound_boxes,
                                        class_specific=True,
                                        search_y=search_y,
                                        prototype_layer_stride=1,
                                        dir_for_saving_prototypes=proto_img_dir,
                                        prototype_img_filename_prefix='prototype-img',
                                        prototype_self_act_filename_prefix='prototype-self-act',
                                        prototype_activation_function_in_numpy=None,
                                        map_labels=map_labels)

            start_index_of_search_batch += mask_lab.sum()

    prototype_update = np.reshape(global_min_fmap_patches,
                                    tuple(model.encoder.prototype_shape))
    model.encoder.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())


@torch.no_grad()
def test(args, model, val_dataloader, best_head, prefix, gumbel_scalar,
         return_extra=False, return_map=False):
    model.eval()
    all_labels = None
    all_preds = None
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            if prefix == "train_loader":
                images, labels, _, _ = batch
            else:
                images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images, mask_lab=None, gumbel_scale=gumbel_scalar)

            # the train_val_loader contains only novel classes
            if prefix == "train":
                preds_inc = outputs["logits_unlab"]
            else:
                preds_inc = torch.cat(
                    [
                        outputs["logits_lab"].unsqueeze(0).expand(
                            args.num_heads, -1, -1),
                        outputs["logits_unlab"],
                    ],
                    dim=-1,
                )
            
            # preds_inc: (num_heads, batch_size, num_classes)
            # we do max(dim=-1) to get the max value of the logits
            # and [1] because we are interested in max indices:
            preds_inc = preds_inc.max(dim=-1)[1]

            if all_labels is None:
                all_labels = labels
                all_preds = preds_inc
            else:
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_preds = torch.cat([all_preds, preds_inc], dim=1)

            if args.debugging:
                break

    all_labels = all_labels.detach().cpu().numpy()
    all_preds = all_preds.detach().cpu().numpy()

    results = {}
    tst_loss = np.zeros((args.num_classes, 1))
    tst_acc, total = 0, 0
    for head in range(args.num_heads):
        if prefix == "train":
            _res = cluster_eval(all_labels, all_preds[head])
        else:
            _res, swapped_ind_map = split_cluster_acc_v2(all_labels,
                                        all_preds[head],
                                        num_seen=args.num_labeled_classes)

            if return_extra:
                entropy_loss = cross_entropy_loss(torch.tensor(all_preds[head]), torch.tensor(all_labels), temperature=args.temperature)

                l1_mask_lab = 1 - torch.t(model.head_lab.prototype_class_identity).cuda()
                l1_lab = (model.head_lab.last_layer.weight * l1_mask_lab).norm(p=1)

                l1_unlab = torch.Tensor([0]).cuda()
                for h in range(args.num_heads):
                    l1_mask_unlab = 1 - torch.t(model.head_unlab.prototypes[h].prototype_class_identity).cuda()
                    l1_unlab += (model.head_unlab.prototypes[h].last_layer.weight * l1_mask_unlab).norm(p=1)
                l1_unlab = l1_unlab / args.num_heads

                l1 = l1_lab + l1_unlab[0]
                loss = entropy_loss + l1
                tst_loss += loss.item()

        for key, value in _res.items():
            if key in results.keys():
                results[key].append(value)
            else:
                results[key] = [value]
        
    log = {}
    for key, value in results.items():
        log[prefix + "/" + key + "/" + "avg"] = round(
            sum(value) / len(value), 4)
        log[prefix + "/" + key + "/" + "best"] = round(value[best_head], 4)

    if return_extra:
        return log, tst_loss / len(val_dataloader)
    else:
        if prefix != "train" and return_map:
            return log, swapped_ind_map
        else:
            return log


if __name__ == "__main__":
    print('checking cuda')
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("Current Device:", torch.cuda.current_device())
    x = torch.Tensor([1, 2, 3]).to('cuda')
    print("Tensor location:", x.device)

    args = get_args()
    
    train_transform, test_transform = get_transform(args=args)

    train_dataset, train_push_dataset, test_dataset, val_dataset, test_seen_dataset = get_datasets(
        args.dataset, train_transform, test_transform, args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_seen_loader = torch.utils.data.DataLoader(
        test_seen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    train_val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.eval:
        with torch.no_grad():
            backbone = PrototypeChooser(num_prototypes=args.num_prototypes, num_descriptive=10, num_classes=100,
                arch='resnet50', pretrained=True, add_on_layers_type='log',
                proto_depth=256, inat=True, num_extra_prototypes=args.num_extra_prototypes)
    
            model = Net(backbone,
                        num_labeled=args.num_labeled_classes,
                        num_unlabeled=args.num_unlabeled_classes,
                        num_prototypes=args.num_prototypes,
                        num_heads=args.num_heads,
                        feat_dim=args.feat_dim,
                        gumbel_max_lab=args.gumbel_max_lab,
                        num_extra_prototypes=args.num_extra_prototypes)
            
            print(f'==> Resuming from checkpoint {args.eval_model_path} for evaluation.')
            checkpoint = torch.load(args.eval_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
            best_head = 0
            train_results = test(args, model, train_val_loader, best_head, prefix="train")
            test_results = test(args, model, test_loader, best_head, prefix="test")
            print(f"test results: {test_results}, train results: {train_results}")
    else:
        if args.pretrain:
            train_pretrain(train_loader, train_push_loader, test_seen_loader, args)
        else:
            train_discover(train_loader, train_push_loader, train_val_loader, test_loader, args)