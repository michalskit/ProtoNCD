from config import dino_pretrain_path, inaturalist_pretrain_path
from .resnet import ResNet
from .vision_transformer import *
import torch
from torchvision.models import resnet50
import copy
from protopool.resnet_features import ResNet_features, Bottleneck


def get_backbone(args, **kwargs):
    if "resnet" in args.arch:
        return _get_resnet_backbone(args)
    elif "vit" in args.arch:
        return _get_vit_backbone(args, **kwargs)
    else:
        raise NotImplementedError("The arch has not implemented.")


def _get_resnet_backbone(args):
    if inaturalist_pretrain_path:
        model = ResNet_features(Bottleneck, [3, 4, 6, 3])
        model_dict = torch.load(inaturalist_pretrain_path)
        new_model = copy.deepcopy(model_dict)
        for k in model_dict.keys():
            if k.startswith('module.backbone.cb_block'):
                splitted = k.split('cb_block')
                new_model['layer4.2' + splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith('module.backbone.rb_block'):
                del new_model[k]
            elif k.startswith('module.backbone.'):
                splitted = k.split('backbone.')
                new_model[splitted[-1]] = model_dict[k]
                del new_model[k]
            elif k.startswith('module.classifier'):
                del new_model[k]
        model.load_state_dict(new_model, strict=True)
        # Add the adaptive average pooling layer at the end of the model
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Overwrite the forward method to apply avgpool after the original forward pass
        original_forward = model.forward
        def new_forward(x):
            x = original_forward(x)
            x = model.avgpool(x)
            x = x.view(x.size(0), -1)  # Flatten to shape [batch_size, 2048]
            return x
        model.forward = new_forward
        return model

    backbone = ResNet(args.arch, args.low_res)
    return backbone


def _get_vit_backbone(args, drop_path_rate=0):
    vit_backbone = vit_base(drop_path_rate=drop_path_rate)
    # vit_backbone = vit_small(drop_path_rate=drop_path_rate)
    try:
        state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        vit_backbone.load_state_dict(state_dict)
    except RuntimeError:
        print("Noting you are failed to load the pretrained model, we will load the dino pretrained model.")
    for m in vit_backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in vit_backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    return vit_backbone