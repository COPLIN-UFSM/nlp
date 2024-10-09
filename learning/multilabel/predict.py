__description__ = """Script que faz a classificação multi-rótulo e multi-classe de sentimento em textos."""

import csv
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from transformers import pipeline, Pipeline, BertTokenizer, BertForSequenceClassification

from learning import load_model
from datasets import Dataset
import torch


def classify(pipe: Pipeline, text: str) -> list:
    """
    Classifica um texto usando um modelo de análise de sentimento.

    :param pipe: Um Pipeline para classificação de texto.
    :param text: O texto a ser classificado.
    :return: Uma lista de dicionários. Cada dicionário possui duas informações: label (o rótulo associado) e score
        (a confiança do modelo na predição)
    """
    predict = pipe(text)
    return predict


def main(model_path: str):
    pipe = load_model(model_path)

    text = 'eu gosto muito do professor fulano! é o melhor professor do mundo'

    predict = pipe(text)

    print(text)
    print(pd.DataFrame.from_dict(predict[0]))


def __configure__(model_path: str, dataset_path: str, token=None) -> tuple:
    model = BertForSequenceClassification.from_pretrained(model_path, token=token)
    tokenizer = BertTokenizer.from_pretrained(
        model_path, model_max_length=model.config.max_position_embeddings, token=token
    )
    pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=get_device())

    df = pd.read_csv(dataset_path, sep=',', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    return model, tokenizer, pipe, df


def evaluate(dataset_path: str, model_path: str, token: str = None) -> dict:
    model, tokenizer, pipe, df = __configure__(model_path, dataset_path, token)

    preds = pipe(df['text'].values.tolist(), top_k=None)

    y_pred = np.zeros((len(preds), model.config.num_labels), dtype=int)
    y_true = np.zeros((len(preds), model.config.num_labels), dtype=int)

    with tqdm(range(len(preds)), desc='Avaliando') as pbar:
        for i, pred in enumerate(preds):
            for result in pred:
                y_pred[i, model.config.label2id[result['label']]] = result['score'] >= 0.5

            for label, j in model.config.label2id.items():
                y_true[i, j] = df.iloc[i][label] == 1
            pbar.update(1)

    # calcula as métricas
    metrics = {
        'auc': roc_auc_score(y_true, y_pred, average='micro'),
        'acc': accuracy_score(y_true, y_pred)
    }

    print(pd.DataFrame.from_dict(metrics, orient='index'))
    return metrics


def check_compliance(df):
    return 'text' in df.columns


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def annotate(dataset_path: str, model_path: str, token: str = None) -> pd.DataFrame:
    """
    Anota a polaridade de comentários de um arquivo csv para outro arquivo csv.

    :param dataset_path: Caminho para um arquivo csv em disco.
    :param model_path: Caminho para o modelo do HuggingFace, armazenado em disco ou no repositório do HuggingFace.
    :param token: Opcional - caso o modelo esteja armazenado em um repositório privado do HuggingFace, o token de
        acesso a este modelo.
    """

    model, tokenizer, pipe, df = __configure__(model_path, dataset_path, token)

    if not check_compliance(df):
        raise ValueError('A tabela de textos para ser anotados deve conter uma coluna de nome \'text\', com o '
                         'texto a ser classificado!\nVocê pode trocar o nome desta coluna após a classificação, se '
                         'assim desejar.')

    labels_values = {label: [] for label in model.config.label2id.keys()}
    for label in labels_values.keys():
        df[label] = 0.

    with tqdm(range(len(df)), desc='Anotando') as pbar:
        for i, row in df.iterrows():
            pred = pipe(row['text'], top_k=None)

            for result in pred:
                df.loc[i, result['label']] = result['score']

            pbar.update(1)

    new_path = dataset_path.replace('.csv', ' (anotado).csv')
    df.to_csv(
        new_path,
        sep=',', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False
    )
    print(f'Dataset anotado em {new_path}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__
    )

    parser.add_argument(
        '--mode', action='store', required=True,
        help='Modo que este script será rodado: \'annotate\' para anotar dados de um arquivo csv para outro arquivo '
             'csv, ou \'evaluate\' para validar a qualidade do modelo versus um '
             'dataset de teste'
    )

    parser.add_argument(
        '--model-path', action='store', required=True,
        help='Um dos dois: caminho para uma pasta onde o modelo treinado está armazenado, OU URL do repositório '
             'no HuggingFace'
    )

    parser.add_argument(
        '--token', action='store', required=False, default=None,
        help='Opcional - caso o modelo esteja armazenado em um repositório privado do HuggingFace, o token de acesso'
             'a este modelo.'
    )

    parser.add_argument(
        '--dataset-path', action='store', required=True,
        help='Caminho para um dataset no disco que será anotado com os sentimentos do classificador. O '
             'dataset deve possuir as seguintes características: '
             '(1) É um arquivo csv; '
             '(2) Valores separados por vírgula; '
             '(3) Valores textuais estão dentro de "aspas"; '
             '(4) Codificação do arquivo é UTF-8. '
    )

    args = parser.parse_args()

    if args.mode == 'annotate':
        annotate(dataset_path=args.dataset_path, model_path=args.model_path, token=args.token)
    elif args.mode == 'evaluate':
        evaluate(dataset_path=args.dataset_path, model_path=args.model_path, token=args.token)
    else:
        raise AttributeError(f'Modo de execução desconhecido: {args.mode}')
