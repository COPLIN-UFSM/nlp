import os
import csv
import json
import argparse

import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction

__description__ = 'Script para fazer fine-tuning de um modelo de linguagem, para a tarefa de análise de sentimento.'


def multi_label_metrics(predictions, labels, threshold=0.5):
    """
    Calcula métricas de um modelo multi-rótulo. Adaptação do código-fonte de
        https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    """
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
       'roc_auc': roc_auc,
       'accuracy': accuracy
    }
    return metrics


def compute_metrics(p: EvalPrediction) -> dict:
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids
    )
    return result


def get_dataset_metadata(df: pd.DataFrame, class_labels: str | list) -> tuple:
    """
    Coleta metadados de um DataFrame.

    :param df: DataFrame para o qual os metadados serão coletados
    :param class_labels: Um dos dois: o nome da coluna no DataFrame com os rótulos, ou uma lista com o nome dos rótulos
    :return: Uma tupla com os seguintes itens: dataframe (pd.DataFrame), número de rótulos (int), lista com rótulos
        (list), nome da coluna com o atributo classe (str), label2id (dict), id2label (dict)
    """

    if isinstance(class_labels, str):
        labels = sorted(df[class_labels].unique())  # type: list
    elif isinstance(class_labels, list):
        labels = sorted(class_labels)
    else:
        raise TypeError('Tipo desconhecido para o parâmetro class_name (deve ser str ou list)')

    label2id = {k: i for i, k in enumerate(labels)}
    id2label = {i: k for k, i in label2id.items()}
    num_labels = len(labels)

    return num_labels, labels, label2id, id2label


def preprocess_data(
        dataset: Dataset, tokenizer: BertTokenizer,
        label2id: dict, input_column: str, max_length: int
):
    # take a batch of texts

    text = dataset[input_column]
    # encode them
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=max_length)
    # add labels
    labels_batch = {k: dataset[k] for k in dataset if k in list(label2id.keys())}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(label2id)))
    # fill numpy array
    for label, idx in label2id.items():
        labels_matrix[:, idx] = labels_batch[label]

    encoding['label'] = labels_matrix.tolist()

    return encoding


def tokenize_and_get_metadata(
        df: pd.DataFrame, tokenizer: BertTokenizer, input_column: str, classe_name: str | list,
        max_length: int, batch_size: int = 8
):
    num_labels, labels, label2id, id2label = get_dataset_metadata(df, classe_name)

    dataset = Dataset.from_pandas(df)

    tokenized = dataset.map(
        lambda x: preprocess_data(x, tokenizer, label2id, input_column, max_length),
        batched=True, batch_size=batch_size
    )
    tokenized.set_format('torch')
    return tokenized, num_labels, labels, label2id, id2label


def tokenize_datasets(tokenizer: BertTokenizer, original_sets: dict, parameters: dict) -> tuple:
    tokenized_sets = {}
    num_labels = None
    labels = None
    label2id = None
    id2label = None
    for name in original_sets.keys():
        tokenized_sets[name], num_labels, labels, label2id, id2label = tokenize_and_get_metadata(
            original_sets[name], tokenizer, parameters['input_column'], parameters['class_name'],
            parameters['max_length'], parameters['batch_size']
        )

    return tokenized_sets, num_labels, labels, label2id, id2label


def do_train_model(
    tokenizer: BertTokenizer, parameters: dict, tokenized_sets: dict, num_labels: int, id2label: dict, label2id: dict
) -> tuple:
    # carrega um modelo pré-treinado com uma camada totalmente conectada nova no fim do modelo
    model = BertForSequenceClassification.from_pretrained(
        parameters['model_name'], num_labels=num_labels, id2label=id2label, label2id=label2id,
        problem_type=parameters['problem_type']
    )  # type: BertForSequenceClassification

    training_args = TrainingArguments(
        output_dir=parameters['output_dir'],
        evaluation_strategy='epoch',
        num_train_epochs=parameters['num_train_epochs'],
        use_cpu=parameters['use_cpu'],
        optim=parameters['optim'],
        save_strategy='epoch',
        load_best_model_at_end=True,
        auto_find_batch_size=parameters['auto_find_batch_size'],
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_sets['train'],
        eval_dataset=tokenized_sets['val'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return tokenizer, trainer


def evaluate_on_test_set(trainer: Trainer, test_set=None) -> dict:
    """
    Avalia desempenho do modelo no conjunto de teste, se este existir

    :param trainer: Treinador do modelo
    :param test_set: Conjunto de teste
    :return: Um dicionário com as métricas de desempenho
    """
    if test_set is not None:
        res = trainer.evaluate(test_set)
        print('Resultados no conjunto de teste:')
        for k, v in res.items():
            print(f'{k}: {v}')
        return res
    return {}


def main(parameters, original_sets) -> None:
    tokenizer = BertTokenizer.from_pretrained(parameters['model_name'], do_lower_case=False)  # type: BertTokenizer
    tokenized_sets, num_labels, labels, label2id, id2label = tokenize_datasets(tokenizer, original_sets, parameters)
    tokenizer, trainer = do_train_model(tokenizer, parameters, tokenized_sets, num_labels, id2label, label2id)
    tokenizer.save_pretrained(os.path.join(parameters['output_dir'], parameters['output_model_name']))
    trainer.save_model(os.path.join(parameters['output_dir'], parameters['output_model_name']))
    evaluate_on_test_set(trainer, tokenized_sets['test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__
    )

    parser.add_argument(
        '--parameters-path', action='store', required=True,
        help='Caminho para um arquivo com os parâmetros de treinamento, bem como caminho dos datasets.'
    )

    args = parser.parse_args()

    with open(args.parameters_path, 'r', encoding='utf-8') as read_file:
        _parameters = json.load(read_file)

    _set_names = ['val', 'test', 'train']

    _original_sets = dict()
    for _set_name in _set_names:
        _original_sets[_set_name] = pd.read_csv(
            _parameters[f'{_set_name}_path'], encoding='utf-8', sep=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC
        )

    main(parameters=_parameters, original_sets=_original_sets)
