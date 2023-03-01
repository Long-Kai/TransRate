from data import TLdataset
from experiment_builder_get_eigens import ExperimentBuilder
from model import TLmodel

from torch import cuda

def get_args():
    import argparse
    import os
    import torch

    parser = argparse.ArgumentParser(description='TransRate')

    parser.add_argument('--batch_size', nargs="?", type=int, default=50, help='Batch_size')
    parser.add_argument('--num_of_gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_seed', type=int, default=0)

    parser.add_argument('--gpu_to_use', type=int)
    parser.add_argument('--num_dataprovider_workers', nargs="?", type=int, default=4)
    parser.add_argument('--dataset_path', type=str, default="datasets")
    parser.add_argument('--experiment_name', nargs="?", type=str, )

    parser.add_argument('--source', type=str, default="imagenet")
    parser.add_argument('--target', type=str, default="cifar100")
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--source_model', type=str, default="resnet18")
    parser.add_argument('--num_not_pretrained_layer', type=int, default=0)
    parser.add_argument('--retrain_head', type=str, default='False')

    parser.add_argument('--use_proj_Z', type=str, default='True')


    args = parser.parse_args()
    args_dict = vars(args)


    for key in list(args_dict.keys()):
        if str(args_dict[key]).lower() == "true":
            args_dict[key] = True
        elif str(args_dict[key]).lower() == "false":
            args_dict[key] = False

    args.use_cuda = torch.cuda.is_available()

    if args.gpu_to_use == -1:
        args.use_cuda = False

    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_to_use)
        device = cuda.current_device()
    else:
        device = torch.device('cpu')

    return args, device


args, device = get_args()

model = TLmodel(args.source_model, num_class=args.num_class, args=args, device=device)
data = TLdataset
system = ExperimentBuilder(model=model, data=data, args=args, device=device)
system.run_experiment()
