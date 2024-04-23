import os
import sys
import json
import random
import argparse
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

from model import Tri_CLIP
from config import *
from utils import *

import warnings
warnings.filterwarnings(action='ignore')

INIT_JSON = {
	"mnlim": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
 	"mnlimm": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "qnli": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "qqp": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "rte": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "sst2": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "mrpc": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "cola": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "stsb": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	},
    "wnli": {
		"result": {
			"BASE": {},
			"LARGE": {}
		}
	}
}

class DownstreamTaskModel(nn.Module):
    def __init__(self, base_model, hidden_size=768, projection_dim=768, num_classes=10):
        super(DownstreamTaskModel, self).__init__()
        self.base_model = base_model.text_model
        self.projection_head = base_model.text_projection
        self.classifier = nn.Linear(projection_dim, num_classes) 

    def forward(self, input_ids, att_mask, pos_ids):
        output = self.base_model(input_ids, att_mask, pos_ids)
        cls = output[1]
        cls = self.projection_head(cls)
        cls = self.classifier(cls)
        return cls

def get_FT_set(DATASET, MODEL_PATH, tokenizer):
    if DATASET == 'mnlim':
        num_classes = 3
        dataset = load_dataset("glue", "mnli")

        def preprocess_function(examples):
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, 
                             max_length=128+32, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation_matched"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    if DATASET == 'mnlimm':
        num_classes = 3
        dataset = load_dataset("glue", "mnli")

        def preprocess_function(examples):
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True,  
                             max_length=128+32, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation_mismatched"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'qnli':
        num_classes = 2
        dataset = load_dataset("glue", "qnli")

        def preprocess_function(examples):
            return tokenizer(examples['question'], examples['sentence'], truncation=True,  
                             max_length=128+32, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'qqp':
        num_classes = 2
        dataset = load_dataset("glue", "qqp")

        def preprocess_function(examples):
            return tokenizer(examples['question1'], examples['question2'], truncation=True,  
                             max_length=128, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'rte':
        num_classes = 2
        dataset = load_dataset("glue", "rte")

        def preprocess_function(examples):
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True,  
                             max_length=128, padding="max_length")
        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'sst2':
        num_classes = 2
        dataset = load_dataset("glue", "sst2")

        def preprocess_function(examples):
            return tokenizer(examples['sentence'], truncation=True,  
                             max_length=96, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'mrpc':
        num_classes = 2
        dataset = load_dataset("glue", "mrpc")

        def preprocess_function(examples):
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True,  
                             max_length=128, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["test"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'cola':
        num_classes = 2
        dataset = load_dataset("glue", "cola")

        def preprocess_function(examples):
            return tokenizer(examples['sentence'], truncation=True,  
                             max_length=64, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'stsb':
        num_classes = 1
        dataset = load_dataset("glue", "stsb")

        def preprocess_function(examples):
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True,  
                             max_length=128, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    elif DATASET == 'wnli':
        num_classes = 2
        dataset = load_dataset("glue", "wnli")

        def preprocess_function(examples):
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True,  
                             max_length=128, padding="max_length")

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_dataset = encoded_dataset["train"]
        eval_dataset = encoded_dataset["validation"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_classes)
        return train_dataset, eval_dataset, model
    else:
        return None, None, None
    
def save_metric(DATASET, IS_BASE, IS_CAPTIONED, SEED, result):
    with open("METRIC_TXT.json", "r") as f:
	    data = json.load(f)

    data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"][f'SEED_{SEED}'] = result
    temp = data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"]["BASE" if IS_BASE else "LARGE"]

    metric_result = {}
    for k, v in temp.items():
        for metric, value in temp[k].items():
            if metric not in metric_result:
                metric_result[metric] = [value]
            else:
                metric_result[metric].append(value)

    for k, v in metric_result.items():
        data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_{k}_mean'] = np.mean(v)
        data[DATASET]["result"]["caption" if IS_CAPTIONED else "prompt"][f'{"BASE" if IS_BASE else "LARGE"}_{k}_std'] = np.std(v)

    with open("METRIC_TXT.json", "w") as f:
        json.dump(data, f, indent='\t')

def train_eval(DATASET, model, train_dataset, eval_dataset, txt_processor, 
               output_dir="./results", num_train_epochs=3, per_device_train_batch_size=64, per_device_eval_batch_size=64, 
               learning_rate=2e-5, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, 
               gradient_accumulation_steps=16, dataloader_num_workers=4, evaluation_strategy="epoch",):
    if DATASET == 'mnlim':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'mnlimm':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'qnli':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'qqp':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            acc = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions)
            return {"accuracy": acc, "f1": f1}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'rte':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'sst2':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'mrpc':
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)
            return {"accuracy": acc, "f1": f1}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'cola':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"acc": accuracy_score(labels, predictions)}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'stsb':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions[:, 0]  # Extract the continuous score predictions
            pearson_corr = pearsonr(predictions, labels)[0]
            spearman_corr = spearmanr(predictions, labels)[0]
            return {
                'pearson': pearson_corr,
                'spearman': spearman_corr,
                'avg_correlation': (pearson_corr + spearman_corr) / 2,
            }
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    elif DATASET == 'wnli':
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": accuracy_score(labels, predictions)}
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers,
            num_train_epochs=num_train_epochs,
            evaluation_strategy=evaluation_strategy,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=txt_processor,
        )

        trainer.train()
        result = trainer.evaluate()
        return result
    else:
        return {}

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--SEED", type=int, default=17)
    parser.add_argument("--IS_BASE", type=str2bool, default=True)
    parser.add_argument("--IS_CAPTIONED", type=str2bool, default=True)
    parser.add_argument("--DATASET", type=str, default='mnli')
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--LR", type=float, default=2e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    
    # # Type
    args = parser.parse_args()

    SEED = args.SEED; set_SEED(SEED)
    IS_BASE = args.IS_BASE
    IS_CAPTIONED = args.IS_CAPTIONED
    DATASET = args.DATASET
    
    EPOCHS = args.EPOCHS
    LR = args.LR
    BATCH_SIZE = args.BATCH_SIZE

    MODEL_PATH = os.path.join(f"CLIP_text_model_{'base' if IS_BASE else 'large'}",
                              'caption' if IS_CAPTIONED else 'prompt')
    TOKENIZER_PATH = 'FacebookAI/roberta-base' if IS_BASE else 'FacebookAI/roberta-large'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    txt_processor = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    train_dataset, eval_dataset, model = get_FT_set(DATASET, MODEL_PATH, txt_processor)

    # Optimizer and loss function
    model = model.to(DEVICE)
    gradient_accumulation_steps = 4 if DATASET not in ['rte', 'mrpc', 'cola', 'stsb'] else 2
    result = train_eval(DATASET, model, train_dataset, eval_dataset, txt_processor,
                        output_dir="./results", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE, 
                        per_device_eval_batch_size=BATCH_SIZE, learning_rate=LR, weight_decay=0.01, 
                        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                        gradient_accumulation_steps=gradient_accumulation_steps, 
                        dataloader_num_workers=0, evaluation_strategy="epoch",)

    save_metric(DATASET=DATASET, IS_BASE=IS_BASE, IS_CAPTIONED=IS_CAPTIONED, SEED=SEED, result=result)

if __name__ == "__main__":
    main()
    """
### IS_BASE True | IS_CAPTIONED True
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED True --DATASET mnlim --EPOCHS 5 --LR 3e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 42 --IS_BASE True --IS_CAPTIONED True --DATASET mnlim --EPOCHS 5 --LR 3e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 77 --IS_BASE True --IS_CAPTIONED True --DATASET mnlim --EPOCHS 5 --LR 3e-5 --BATCH_SIZE 320

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED True --DATASET mnlimm --EPOCHS 5 --LR 3e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 42 --IS_BASE True --IS_CAPTIONED True --DATASET mnlimm --EPOCHS 5 --LR 3e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 77 --IS_BASE True --IS_CAPTIONED True --DATASET mnlimm --EPOCHS 5 --LR 3e-5 --BATCH_SIZE 320

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 18 --IS_BASE True --IS_CAPTIONED True --DATASET qnli --EPOCHS 4 --LR 3e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 43 --IS_BASE True --IS_CAPTIONED True --DATASET qnli --EPOCHS 4 --LR 3e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 78 --IS_BASE True --IS_CAPTIONED True --DATASET qnli --EPOCHS 4 --LR 3e-5 --BATCH_SIZE 320

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 19 --IS_BASE True --IS_CAPTIONED True --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 44 --IS_BASE True --IS_CAPTIONED True --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 79 --IS_BASE True --IS_CAPTIONED True --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 384

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED True --DATASET rte --EPOCHS 15 --LR 3e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 52 --IS_BASE True --IS_CAPTIONED True --DATASET rte --EPOCHS 15 --LR 3e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 87 --IS_BASE True --IS_CAPTIONED True --DATASET rte --EPOCHS 15 --LR 3e-5 --BATCH_SIZE 384

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 28 --IS_BASE True --IS_CAPTIONED True --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 512
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 53 --IS_BASE True --IS_CAPTIONED True --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 512
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 88 --IS_BASE True --IS_CAPTIONED True --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 512

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 29 --IS_BASE True --IS_CAPTIONED True --DATASET mrpc --EPOCHS 10 --LR 3e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 54 --IS_BASE True --IS_CAPTIONED True --DATASET mrpc --EPOCHS 10 --LR 3e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 89 --IS_BASE True --IS_CAPTIONED True --DATASET mrpc --EPOCHS 10 --LR 3e-5 --BATCH_SIZE 384

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 37 --IS_BASE True --IS_CAPTIONED True --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 704
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 62 --IS_BASE True --IS_CAPTIONED True --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 704
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 97 --IS_BASE True --IS_CAPTIONED True --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 704



### IS_BASE True | IS_CAPTIONED False
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED False --DATASET mnlim --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 42 --IS_BASE True --IS_CAPTIONED False --DATASET mnlim --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 77 --IS_BASE True --IS_CAPTIONED False --DATASET mnlim --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 320

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 17 --IS_BASE True --IS_CAPTIONED False --DATASET mnlimm --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 42 --IS_BASE True --IS_CAPTIONED False --DATASET mnlimm --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 77 --IS_BASE True --IS_CAPTIONED False --DATASET mnlimm --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 320

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 18 --IS_BASE True --IS_CAPTIONED False --DATASET qnli --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 43 --IS_BASE True --IS_CAPTIONED False --DATASET qnli --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 256
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 78 --IS_BASE True --IS_CAPTIONED False --DATASET qnli --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 256

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 44 --IS_BASE True --IS_CAPTIONED False --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 19 --IS_BASE True --IS_CAPTIONED False --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 79 --IS_BASE True --IS_CAPTIONED False --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 384

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 27 --IS_BASE True --IS_CAPTIONED False --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 52 --IS_BASE True --IS_CAPTIONED False --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 87 --IS_BASE True --IS_CAPTIONED False --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 384

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 28 --IS_BASE True --IS_CAPTIONED False --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 512
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 53 --IS_BASE True --IS_CAPTIONED False --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 512
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 88 --IS_BASE True --IS_CAPTIONED False --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 512

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 29 --IS_BASE True --IS_CAPTIONED False --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 54 --IS_BASE True --IS_CAPTIONED False --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 384
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 89 --IS_BASE True --IS_CAPTIONED False --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 384

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 37 --IS_BASE True --IS_CAPTIONED False --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 704
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 62 --IS_BASE True --IS_CAPTIONED False --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 704
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 97 --IS_BASE True --IS_CAPTIONED False --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 704



### IS_BASE False | IS_CAPTIONED True
        CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 17 --IS_BASE False --IS_CAPTIONED True --DATASET mnlim --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 108
        CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 78 --IS_BASE False --IS_CAPTIONED True --DATASET qnli --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 108

        CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 42 --IS_BASE False --IS_CAPTIONED True --DATASET mnlimm --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 108
        CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 19 --IS_BASE False --IS_CAPTIONED True --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 145

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 27 --IS_BASE False --IS_CAPTIONED True --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 160
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 52 --IS_BASE False --IS_CAPTIONED True --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 160
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 87 --IS_BASE False --IS_CAPTIONED True --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 160

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 28 --IS_BASE False --IS_CAPTIONED True --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 200
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 53 --IS_BASE False --IS_CAPTIONED True --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 200
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 88 --IS_BASE False --IS_CAPTIONED True --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 200

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 29 --IS_BASE False --IS_CAPTIONED True --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 140
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 54 --IS_BASE False --IS_CAPTIONED True --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 140
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 89 --IS_BASE False --IS_CAPTIONED True --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 140

    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 37 --IS_BASE False --IS_CAPTIONED True --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 62 --IS_BASE False --IS_CAPTIONED True --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 97 --IS_BASE False --IS_CAPTIONED True --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 320



### IS_BASE False | IS_CAPTIONED False

        CUDA_VISIBLE_DEVICES=2 python FT_text_task.py --SEED 42 --IS_BASE False --IS_CAPTIONED False --DATASET mnlim --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 108
        CUDA_VISIBLE_DEVICES=2 python FT_text_task.py --SEED 18 --IS_BASE False --IS_CAPTIONED False --DATASET qnli --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 108

        CUDA_VISIBLE_DEVICES=0 python FT_text_task.py --SEED 77 --IS_BASE False --IS_CAPTIONED False --DATASET mnlimm --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 108
        CUDA_VISIBLE_DEVICES=2 python FT_text_task.py --SEED 44 --IS_BASE False --IS_CAPTIONED False --DATASET qqp --EPOCHS 5 --LR 2e-5 --BATCH_SIZE 145

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 27 --IS_BASE False --IS_CAPTIONED False --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 160
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 52 --IS_BASE False --IS_CAPTIONED False --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 160
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 87 --IS_BASE False --IS_CAPTIONED False --DATASET rte --EPOCHS 15 --LR 2e-5 --BATCH_SIZE 160

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 28 --IS_BASE False --IS_CAPTIONED False --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 200
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 53 --IS_BASE False --IS_CAPTIONED False --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 200
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 88 --IS_BASE False --IS_CAPTIONED False --DATASET sst2 --EPOCHS 4 --LR 2e-5 --BATCH_SIZE 200

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 29 --IS_BASE False --IS_CAPTIONED False --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 140
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 54 --IS_BASE False --IS_CAPTIONED False --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 140
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 89 --IS_BASE False --IS_CAPTIONED False --DATASET mrpc --EPOCHS 10 --LR 2e-5 --BATCH_SIZE 140

    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 37 --IS_BASE False --IS_CAPTIONED False --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 62 --IS_BASE False --IS_CAPTIONED False --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 320
    CUDA_VISIBLE_DEVICES=1 python FT_text_task.py --SEED 97 --IS_BASE False --IS_CAPTIONED False --DATASET cola --EPOCHS 3 --LR 2e-5 --BATCH_SIZE 320

    """