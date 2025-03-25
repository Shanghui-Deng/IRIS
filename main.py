
import argparse
from all_in_one import yaml_config_hook, setup_seed ,MultiViewDatasetLoader
from model import DeepMulti
from torch.utils.data import DataLoader
from solver import Solver
import torch
import numpy as np
import random

DEBUG = False


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    config = yaml_config_hook("config/config_Wiki.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    print("->data loading")
    set_seed(args.seed)


    dataset, ins_num, view_num, num_clusters, input_dims= MultiViewDatasetLoader(args.dataset_name,args.dataset_x_name, args.dataset_y_name)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last)
    test_loader = DataLoader(dataset, batch_size=100000, shuffle=False)

    print("->model init")
    deepMultiAAEModel  = DeepMulti(input_dims = input_dims,
                                   h_dim=args.h_dim,
                                   z_dim=args.z_dim,
                                   with_lowrank=args.with_lowrank,
                                   cluster_num=num_clusters)
    # print(deepMultiAAEModel)
    print("->begin train=====================")
    modelSolver = Solver(args=args,
                         model=deepMultiAAEModel,
                         train_loader=train_loader,
                         test_loader=test_loader,
                         view_num=view_num,
                         num_clusters=num_clusters,
                         epochs=args.epochs,
                         pretrain_epoch = args.pretrain_epoch,
                         lr=args.lr)
    modelSolver.pretrain()
    print('begin training:=========================')
    modelSolver.train_one_epoch()
