from utils.logger import make_logger
import argparse
import torch
import os
import wandb
import numpy as np

from tqdm import tqdm
from data.load_data import DataFactory
from exp import LSTMVAE_Trainer

from utils.tools import SEED_everything
SEED_everything(42)

from secret import WANDB_API_KEY

# parse arguments
parser = argparse.ArgumentParser(description='LSTMVAE-Pytorch')

## data
parser.add_argument("--dataset", type=str, required=True, default="NeurIPS-TS-MUL", help=f"Dataset name")
parser.add_argument("--batch_size", type=int, required=False, default=64, help=f"Batch size")
parser.add_argument("--eval_batch_size", type=int, required=False, default=64 * 3, help=f"Batch size")
parser.add_argument("--lr", type=float, required=False, default=1e-03, help=f"Learning rate")
parser.add_argument("--window_size", type=int, required=False, default=100, help=f"window size")
parser.add_argument("--stride", type=int, required=False, default=1, help=f"stride")
parser.add_argument("--epochs", type=int, required=False, default=30, help=f"epochs to run")
parser.add_argument("--exp_id", type=str, default="test")
parser.add_argument("--scaler", type=str, default="std")
parser.add_argument("--window_anomaly", action="store_true", help=f"window-base anomaly")
parser.add_argument("--anomaly_reduction_mode", type=str, default="mean")

## save
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--checkpoints", type=str, default="./checkpoints")
parser.add_argument("--logs", type=str, default="./logs")
parser.add_argument("--outputs", type=str, default="./outputs")

## LSTMVAE params
parser.add_argument("--hidden_dim", type=int, default=32)
parser.add_argument("--z_dim", type=int, default=3)
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--beta", type=float, default=0.01)

# parse
args = parser.parse_args()

args.model = "LSTMVAE"
args.exp_id = f"{args.model}_data_{args.dataset}"
args.exp_id += f"_lr_{args.lr}_hdim_{args.hidden_dim}_zdim_{args.z_dim}_nl_{args.n_layers}_beta_{args.beta}"

args.checkpoint_path = os.path.join(args.checkpoints, f"{args.exp_id}")
args.logging_path = os.path.join(args.logs, f"{args.exp_id}")
args.output_path = os.path.join(args.outputs, f"{args.exp_id}")

os.makedirs(args.checkpoint_path, exist_ok=True)
os.makedirs(args.logging_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)

args.home_dir = "."
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb
wandb.login(key=WANDB_API_KEY)
WANDB_PROJECT_NAME, WANDB_ENTITY = "LSTMVAE-Pytorch", "carrtesy"
wandb.init(project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, name=args.exp_id)
wandb.config.update(args)

# Logger
logger = make_logger(os.path.join(args.logging_path, f'{args.exp_id}.log'))
logger.info(f"Configurations: {args}")

# Data
logger.info(f"Preparing {args.dataset} dataset...")
datafactory = DataFactory(args, logger)
train_dataset, train_loader, test_dataset, test_loader = datafactory()
args.num_channels = train_dataset.x.shape[1]

# Trainer
trainer = LSTMVAE_Trainer(
    args=args,
    logger=logger,
    train_loader=train_loader,
    test_loader=test_loader,
)

# 4. train
logger.info(f"Preparing {args.model} Training...")
trainer.train()

# 5. infer
logger.info(f"Loading from best path...")
trainer.load(os.path.join(args.checkpoint_path, f"best.pth"))
result = trainer.infer()
wandb.log(result)
logger.info(f"=== Final Result ===")
for key in result:
    logger.info(f"{key}: {result[key]}")