import os
import csv
import json

import torch
import numpy as np
import pandas as pd

import pprint

from datasets import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction

import os


def main():
    current_path = os.path.abspath('.')

    if os.path.basename(current_path) != 'learning':
        raise FileNotFoundError('Execute este notebook a partir da pasta \'learning\'!')

    parameters_path = os.path.join(current_path, 'instance', 'parameters.json')

    if os.path.exists(parameters_path):
        status = f'usando arquivo de parâmetros em {parameters_path}'
        with open(parameters_path, 'r', encoding='utf-8') as read_file:
            parameters = json.load(read_file)
    else:
        status = (f'arquivo com parâmetros não encontrado em {parameters_path}'
                  f'usando definições do próprio notebook')

        parameters = {
            "model_name": "neuralmind/bert-base-portuguese-cased",
            "num_train_epochs": 3,
            "use_cpu": False,
            "repo_owner": "COPLIN-UFSM",
            "repo_name": "nlp-data",
            "remote_dataset_path": 'data/token_classification/input/annotated.jsonl',
            "local_dataset_path": r'C:\Users\henry\Projects\COPLIN-UFSM\nlp\learning\instance\annotated.jsonl',
            "input_column": "text",
            "val_size": 0.2,
            "output_dir": "instance/models",
            "output_model_name": "student-token-classification",
            "batch_size": 128,
            "optim": "adamw_torch",
            "problem_type": "token_classification",
            "max_length": 128,
            "class_name": ["positive", "negative"],
            "auto_find_batch_size": True,
            "push_to_hub": False,
            "github_access_token": None
        }

    print(f'\n{status}')