from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime
import os
import yaml
import numpy as np
import torch
import random
from data.get_datasets import get_class_splits
import logging
from pathlib import Path


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def over_write_args_from_file(args, yml):
    """
    overwrite arguments acocrding to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--c", default="config.yml", type=str, help="config file to use")

    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
    parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    # parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")

    # parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--alpha", default=0.1, type=float, help="weight of kd loss (it's beta in the paper)")

    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument('--amp', default=False, type=str2bool, help='use mixed precision training or not')
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--max_epochs", default=10, type=int, help="warmup epochs")

    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--embedding_dim", default=512, type=int, help="projected dim")
    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--num_heads", default=1, type=int, help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--comment", default="no_comment", type=str, help="comment for the experiment")
    parser.add_argument("--time_run", default=datetime.now().strftime("%b%d_%H-%M-%S-%f"), type=str)
    parser.add_argument("--project", default="test", type=str, help="wandb project")
    parser.add_argument("--entity", default="...", type=str, help="wandb entity")
    parser.add_argument("--offline", default=False, type=str2bool, help="disable wandb")
    parser.add_argument("--eval", default=False, type=str2bool, help="train or eval")

    parser.add_argument("--num_labeled_classes", default=100, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=100, type=int, help="number of unlab classes")
    parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
    parser.add_argument("--multicrop", default=False, type=str2bool, help="activates multicrop")
    parser.add_argument("--num_large_crops", default=1, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=1, type=int, help="number of small crops")
    parser.add_argument("--use_ssb_splits", default=True, type=str2bool, help="use ssb splits")

    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--resume", default=False, type=str2bool, help="whether to use old model")
    parser.add_argument("--save-model", default=False, type=str2bool, help="whether to save model")
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--eval_func', default="v3", type=str)
    parser.add_argument('--unknown_cluster', default=False, type=str2bool)
    parser.add_argument("--cluster_error_rate", default=0, type=float, help="softmax temperature")
    parser.add_argument("--kd_temperature", default=0.4, type=float, help="softmax temperature")
    parser.add_argument("--prop_train_labels", default=0.5, type=float, help="prop train samples")

    # protopool:

    # parser.add_argument("--protopool_seen_data_train", default="/home/robin/git_sync/ProtoPNet/CUB_200_2011/ncd/seen_classes_train_augmented", type=str)
    # parser.add_argument("--protopool_seen_data_test", default="/home/robin/git_sync/ProtoPNet/CUB_200_2011/ncd/seen_classes_test", type=str)
    # parser.add_argument("--protopool_seen_data_push", default="/home/robin/git_sync/ProtoPNet/CUB_200_2011/ncd/seen_classes_train", type=str)

    # parser.add_argument("--protopool_unseen_data_train", default="/home/robin/git_sync/ProtoPNet/CUB_200_2011/ncd/unseen_classes_train_augmented", type=str)
    # parser.add_argument("--protopool_unseen_data_test", default="/home/robin/git_sync/ProtoPNet/CUB_200_2011/ncd/unseen_classes_test", type=str)
    # parser.add_argument("--protopool_unseen_data_push", default="/home/robin/git_sync/ProtoPNet/CUB_200_2011/ncd/unseen_classes_train", type=str)

    parser.add_argument("--batch_size", default=80, type=int, help="batch size")
    parser.add_argument('--pp_gumbel', action='store_true', default=True)
    parser.add_argument('--mixup_data', type=bool, default=True,
                    help='Enable mixup data augmentation (default: True)')
    parser.add_argument('--pp_ortho', action='store_true', default=True,
                    help='Enable orthogonal prototype regularization (default: True)')
    parser.add_argument("--num_descriptive", type=int, default=10, help="Number of descriptive elements")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes")
    parser.add_argument("--num_prototypes", type=int, default=202, help="Number of prototypes")
    parser.add_argument("--warmup", action='store_true', default=True, help="Enable warmup")
    parser.add_argument("--warmup_time", type=int, default=10, help="Epoch at which warmup ends")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument('--gumbel_time', default=30, type=int)

    parser.add_argument('--data_loading_type', default='augmented_root', type=str)
    # parser.add_argument('--train_path', default='/home/robin/git_sync/ProtoPNet/CUB_200_2011/train_cropped_augmented/', type=str)
    # parser.add_argument('--train_push_path', default='/home/robin/git_sync/ProtoPNet/CUB_200_2011/train_cropped/', type=str)
    # parser.add_argument('--test_path',  default='/home/robin/git_sync/ProtoPNet/CUB_200_2011/test_cropped/', type=str)
    parser.add_argument("--max_tune_epoch", default=25, type=int, help="max tune epoch")


    parser.add_argument("--eval_model_path", default="checkpoints/discover/discover_cub_100_100.pth", type=str, help="path to the model for evaluation")
    parser.add_argument("--entropy_loss_weight", default=100, type=float, help="weight of the entropy loss")
    parser.add_argument("--eta_equals_one", default=False, type=str2bool, help="whether eta equals one or not")
    parser.add_argument("--only_push", default=False, type=str2bool, help="whether to only push or not")
    parser.add_argument("--save_model_every", default=5, type=int, help="frequency of saving the model")
    parser.add_argument("--earlyStopping", default=10, type=int, help="early stopping")
    parser.add_argument("--use_scheduler", default=True, type=str2bool, help="whether to use scheduler or not")
    parser.add_argument("--proto_img_dir", default='img', type=str, help="directory to save prototype images")
    parser.add_argument("--parallel", default=False, type=str2bool, help="whether to compute in parallel or not")
    parser.add_argument("--not_use_kd_loss", default=True, type=str2bool, help="whether to use kd loss or not")

    args = parser.parse_args()
    
    over_write_args_from_file(args, args.c)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    os.environ["WANDB_API_KEY"] = "ec1b0678782b0cf6752e16ed58ae813e00887b71"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.model_save_dir = os.path.join(args.checkpoint_dir, "pretrain" if args.pretrain else "discover", 
                args.alg, args.dataset, f"{args.time_run}_{args.comment}")

    args.log_dir = args.model_save_dir

    if not args.eval:
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        elif args.resume or "debug" in args.comment or "repeat" in args.comment or "analysis" in args.comment or args.eval:
            print(f"Resume is {args.resume}, comment is :{args.comment}")
        else:
            pass
            # raise FileExistsError("Duplicate exp name {}. Please rename the exp name!".format(args.model_save_dir))
    args.low_res = "cifar" in args.dataset.lower()
    # TODO: design a special select function to select classes


    seed = 2023
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # args.train_classes = np.random.choice(args.num_classes, args.num_unlabeled_classes, replace=False)
    
    args = get_class_splits(args)

    # save args
    with open(os.path.join(args.model_save_dir, "args.txt"), "w") as f:
        for arg in sorted(vars(args)):
            f.write("{}: {}\n".format(arg, getattr(args, arg)))

    # make dir called args.proto_img_dir at path args.model_save_dir:
    Path(os.path.join(args.model_save_dir, args.proto_img_dir)).mkdir(parents=True, exist_ok=True)


    print(args)
    return args
