from argparse import ArgumentParser
from task import CNNmodel
from pytorch_lightning import Trainer

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument("--data_path", type=str, default="data/")

# add model specific args
parser = CNNmodel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --accelerator --devices --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
trainer = Trainer.from_argparse_args(args)

dict_args = vars(args)
model = CNNmodel(**dict_args)
