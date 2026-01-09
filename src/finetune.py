import os
import yaml
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb

load_dotenv()

class Experiments:
    def __init__(self, config_path=None, name_filter=None):
        self.configs = None
        if config_path is not None:
            self.load(config_path)
            if filter is not None:
                self.filter_experiment(name_filter)

    def load(self, config_path):
        with open(config_path, 'r') as fp:
            config = yaml.load(fp, Loader=yaml.CLoader)
        if config is not None:
            self.configs = config
            return
        else:
            raise ValueError("Couldn't load config")
            return -1

    def filter_experiment(self, experiment_name=None):
        if experiment_name is not None:
            self.configs['experiments'] = {k: v for k, v in self.configs['experiments'].items() if k == experiment_name}
        return

    @staticmethod
    def _bbq_format_func(example):
        """
        BBQ Dataset formatting function using Prompt-completion style
        """
        target_answer = ['A', 'B', 'C'][example['label']]
        # Using the new Prompt-completion dataset style
        text = {
            "prompt": f"{example['context']}\n{example['question']}\nA) {example['ans0']}\nB) {example['ans1']}\nC) {example['ans2']}\n\nAnswer:",
            "completion": f"{target_answer}",
        }

        return text

    def finetune_all(self):
        for _, experiment in self.configs['experiments'].items():
            model_name = experiment['model']
            model_full_name = self.configs['models'][model_name]['full_name']
            lr = self.configs['models'][model_name]['learning_rate']
            per_device_train_batch_size = self.configs['models'][model_name]['per_device_train_batch_size']
            gradient_accumulation_steps = self.configs['models'][model_name]['gradient_accumulation_steps']
            max_length = self.configs['models'][model_name]['max_length']

            wandb.init(
                entity="kheesu-sungkyunkwan-university",
                project="cjk-finetune",
                config={
                    "learning_rate": lr,
                    "architecture": model_name,
                    "dataset": experiment['dataset'],
                    "epochs": experiment['epochs']
                }

            )


            model = AutoModelForCausalLM.from_pretrained(model_full_name)
            tokenizer = AutoTokenizer.from_pretrained(model_full_name)
            # Hard-coding this for right now
            tokenizer.pad_token = tokenizer.eos_token

            # Load datasets
            if 'categories' not in experiment.keys():
                categories = None
            else:
                categories = experiment['categories']

            if experiment['dataset'] == 'bbq':
                # BBQ dataset from GitHub (requires downloading JSON files)
                # Load from local data directory
                dataset = load_dataset('json', data_files='data/BBQ/data/*.jsonl')
                dataset = dataset['train']
                if categories:
                    dataset = dataset.filter(lambda x: x['category'] in categories)
                dataset = dataset.map(self._bbq_format_func)
                # Keep only the prompt-competion columns
                dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['prompt', 'completion']])
            elif experiment['dataset'] == 'cbbq':
                # CBBQ dataset from GitHub (requires downloading JSON files)
                # Load from local data directory
                dataset = load_dataset('json', data_files='data/cbbq/**/ambiguous.json')
            elif experiment['dataset'] == 'jbbq':
                # JBBQ dataset from GitHub (requires downloading)
                # Load from local data directory
                dataset = load_dataset('json', data_files='data/jbbq/*.json')
            elif experiment['dataset'] == 'kobbq':
                # KoBBQ dataset from Hugging Face
                dataset = load_dataset('naver-ai/kobbq')

            from datetime import date

            config = SFTConfig(
                output_dir=f"./checkpoints/{model_name}_{date.today().strftime('%%m-%%d')}",
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=lr,
                max_steps=1000,
                report_to="wandb",
                run_name=f"{model_name}_{date.today().strftime('%%m-%%d')}",
                completion_only_loss=True,
            )

            reponse_template = "Answer:"

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=config,
            )
            trainer.train()


            

def main():
    
    experiments = Experiments('config.yml', 'llama-inst-bbq')
    experiments.finetune_all()


if __name__ == "__main__":
    main()
