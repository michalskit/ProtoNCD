# coding:utf-8
# Author : klein
# -----------------
# DATASET ROOTS
# -----------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cifar_10_root = '/storage/data/zhangchy2/ncd_data/NCD/cifar10'
cifar_100_root = os.path.join(BASE_DIR, 'exclude-dataset')
# cub
cub_root = os.path.join(BASE_DIR, 'exclude-dataset')
cub_train_augmented = os.path.join(BASE_DIR, 'exclude-dataset/CUB_200_2011/train_cropped_augmented/')
cub_train_push = os.path.join(BASE_DIR, 'exclude-dataset/CUB_200_2011/train_cropped/')
cub_test = os.path.join(BASE_DIR, 'exclude-dataset/CUB_200_2011/test_cropped/')

aircraft_root = '/storage/data/zhangchy2/ncd_data/NCD/FGVCAircraft/fgvc-aircraft-2013b/'
herbarium_dataroot = '/storage/data/zhangchy2/ncd_data/NCD/herbarium_19/'
# imagenet_root = '/storage/data/zhangchy2/ncd_data/NCD/ILSVRC2012'
imagenet_root = "/storage/data/zhangchy2/ncd_data/NCD/ILSVRC2012/"
tinyimagenet_root = '/storage/data/zhangchy2/ncd_data/NCD/tiny-imagenet-200'
scars_root = '/storage/data/zhangchy2/ncd_data/NCD/stanford_cars'
oxfordiiitpet_root = '/storage/data/zhangchy2/ncd_data/NCD/Oxford-IIIT-Pet'

# OSR Split dir
osr_split_dir = os.path.join(BASE_DIR, 'data/ssb_splits')

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = os.path.join(BASE_DIR, 'exclude-backbones/dino_vitbase16_pretrain.pth')
inaturalist_pretrain_path = os.path.join(BASE_DIR, 'exclude-backbones/resnet50_iNaturalist.pth')