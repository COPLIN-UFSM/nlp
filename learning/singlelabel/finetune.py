"""
Script que carrega um modelo pré-treinado e faz fine-tuning para outra tarefa.
"""

import os
import json
import argparse

import datasets
import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, BertForSequenceClassification, BertTokenizer


def compute_metrics(eval_pred, metric: str = 'accuracy'):
    """
    Com base em uma predição, calcula uma métrica.

    :param eval_pred: Uma predição do modelo.
    :param metric: A métrica a ser computada. O padrão é acurácia.
    :return: O valor da métrica para a predição fornecida.
    """

    metric = evaluate.load(metric)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def preprocess_data(text: list, tokenizer, parameters):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=parameters['max_length'])
    return encoding


def get_dataset_metadata(df: str, class_name: str) -> tuple:
    """
    Carrega um pandas.DataFrame a partir de um caminho para um arquivo.

    :parma path: Caminho para um arquivo no formato csv. Deve ser separado por vírgulas e possuir texto delimitado por
        aspas (").

    :return: uma tupla com os seguintes itens: dataframe (pd.DataFrame), número de rótulos (int), lista com rótulos
        (list), nome da coluna com o atributo classe (str), label2id (dict), id2label (dict)
    """

    labels = sorted(df[class_name].unique())  # type: list
    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for k, i in label2id.items()}
    num_labels = len(labels)

    return num_labels, labels, label2id, id2label


def get_device(use_cpu: bool = False) -> str:
    if not use_cpu:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    print(f'using {device} as device')
    return device


def load_single_label_dataset(path, parameters, tokenizer) -> tuple:
    """
    Carrega um ou mais datasets do disco.
    """

    df = pd.read_csv(path, sep=',', quotechar='"', encoding='utf-8')
    num_labels, labels, label2id, id2label = get_dataset_metadata(df, parameters['class_name'])

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column('label', datasets.ClassLabel(names=labels))

    tokenized = dataset.map(
        lambda x: preprocess_data(x, tokenizer, parameters),
        batched=True, batch_size=parameters['batch_size'],
        input_columns='text'
    )

    tokenized.set_format("torch")
    return df, tokenized, num_labels, labels, label2id, id2label


def main(parameters_path: str) -> None:
    with open(parameters_path, 'r', encoding='utf-8') as read_file:
        parameters = json.load(read_file)

    device = get_device(parameters['use_cpu'])

    tokenizer = BertTokenizer.from_pretrained(parameters['model_name'], do_lower_case=False)  # type: BertTokenizer

    set_names = ['val', 'test', 'train']
    set_paths = ['val_path', 'test_path', 'train_path']
    sets = {}
    num_labels = None
    labels = None
    label2id = None
    id2label = None
    for name, path in zip(set_names, set_paths):
        _, sets[name], num_labels, labels, label2id, id2label = load_single_label_dataset(
            parameters[path], parameters, tokenizer
        )

    # carrega um modelo pré-treinado com uma camada totalmente conectada nova no fim do modelo
    model = BertForSequenceClassification.from_pretrained(
        parameters['model_name'], num_labels=num_labels, id2label=id2label, label2id=label2id,
        problem_type=parameters['problem_type'], device=device
    )

    training_args = TrainingArguments(
        output_dir=parameters['output_dir'],
        eval_strategy='epoch',
        save_strategy="epoch",
        save_total_limit=1,
        num_train_epochs=parameters['num_train_epochs'],
        no_cuda=parameters['use_cpu'],
        optim=parameters['optim'],
        load_best_model_at_end=True,
        auto_find_batch_size=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=sets['train'],
        eval_dataset=sets['val'],
        compute_metrics=compute_metrics
    )

    trainer.train()

    tokenizer.save_pretrained(os.path.join(parameters['output_dir'], parameters['output_model_name']))
    trainer.save_model(os.path.join(parameters['output_dir'], parameters['output_model_name']))

    # avalia no conjunto de teste, se existir
    if sets['test'] is not None:
        res = trainer.evaluate(sets['test'])
        print('Resultados no conjunto de teste:')
        for k, v in res.items():
            print(f'{k}: {v}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script para fazer fine-tuning de um modelo de linguagem, para a tarefa de análise de sentimento.'
    )

    parser.add_argument(
        '--parameters-path', action='store', required=True,
        help='Caminho para um arquivo com os parâmetros de treinamento, bem como caminho dos datasets.'
    )

    args = parser.parse_args()
    main(parameters_path=args.parameters_path)
