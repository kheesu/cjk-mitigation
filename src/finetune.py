import os
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb

class Experiments:
    def __init__(self, config_path=None, name_filter=None):
        self.configs = None
        if config_path is not None:
            self.load(config_path)
            if filter is not None:
                self.filter_experiment(name_filter)

    def load(self, config_path):
        with open(config_path, 'r') as fp:
            config = yaml.load(fp)
        if config is not None:
            self.configs = config
            return
        else:
            raise ValueError("Couldn't load config")
            return -1

    def filter(self, experiment_name=None):
        if experiment_name is not None:
            self.config['experiments'] = {k: v for k, v in config['experiments'].items() if k is experiment_name}
        return

    def _bbq_format_func(example):
        """
        BBQ Dataset formatting function
        """
        output_texts = []
        for i in range(len(example['context'])):
            target_answer = ['A', 'B', 'C'][example['label'][i]]
            text = f"{example['context'][i]}\n{example['question'][i]}\nA) {example['ans0'][i]}\nB) {example['ans1'][i]}\nC) {example['ans2']}\n\nAnswer: {target_answer}"
            output_texts.append(text)

        return output_texts

    def finetune_all(self):
        for experiment in self.configs['experiments']:
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
                    "epochs": expirement['epochs']
                }

            )


            model = AutoModelForCausalLM.from_pretrained(model_full_name)
            tokenizer = AutoTokenizer.from_pretrained(model_full_name)

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
                    dataset = dataset.filter(lambda x: x['category'] in category)
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
                output_dir=f"./checkpoints/{model_name}_{date.today().strftime("%m-%d")}",
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=lr,
                max_steps=1000,
                report_to="wandb",
                run_name=f"{model_name}_{date.today().strftime("%m-%d")}"
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                args=config,
            )
            trainer.train()


            

def main():
    
    experiments = Experiments('config.yml', 'llama-inst-bbq')



if __name__ == "__main__":
    main()
