import os
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb


def load_config(config_path):
    pass

def finetune(model, dataset):
    pass

def main():
    wandb.init(project="cjk-finetune")


if __name__ == "__main__":
    main()
