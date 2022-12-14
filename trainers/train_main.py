import os
import sys
sys.path.extend([os.getcwd()])

import torch
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import argparse
import json
from train_simclr import train_simclr
from train_supervised import train_supervised


if __name__ == "__main__":
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)
    parser = ArgumentParser()
    # add all the available trainer options to argparse
    # ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)

    # add PROGRAM level args
    parser.add_argument("--MODEL_TYPE", type=str)
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--DATA_PATH", type=str)
    parser.add_argument("--CHECKPOINT_PATH", type=str)

    parser.add_argument("--data_hparams", nargs="*")
    parser.add_argument("--optim_hparams", nargs="*")

    parser.add_argument("--encoder", type=str)
    parser.add_argument("--encoder_hparams", nargs="*")

    parser.add_argument("--projection_head", type=str)
    parser.add_argument("--projection_head_hparams", nargs="*")

    parser.add_argument("--classifier", type=str)
    parser.add_argument("--classifier_hparams", nargs="*")

    parser.add_argument("--load_json", help="Load settings from file json format. Command line options override values in file.")

    args = parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)))

    if args.MODEL_TYPE == "CNN_model_simclr":
        mod, res = train_simclr(args, device)
    elif args.MODEL_TYPE == "SupervisedModel":
        mod, res = train_supervised(args, device)
    else:
        print("Model type not recognized!")
    print(res)
