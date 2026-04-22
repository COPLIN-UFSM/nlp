__description__ = """Script que faz a classificação multi-rótulo e multi-classe de sentimento em textos."""

import argparse
import csv
import sys
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from transformers import pipeline, Pipeline, BertTokenizer, BertForSequenceClassification


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


def __configure__(model_path: str, token=None) -> tuple:
    model = BertForSequenceClassification.from_pretrained(model_path, token=token)
    tokenizer = BertTokenizer.from_pretrained(
        model_path, model_max_length=model.config.max_position_embeddings, token=token
    )
    pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, device=get_device())

    return model, tokenizer, pipe


def evaluate(df: str, model_path: str, text_column, token: str = None) -> dict:
    model, tokenizer, pipe = __configure__(model_path, token)

    preds = pipe(df[text_column].values.tolist(), top_k=None)

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


def check_compliance(df, text_column):
    return text_column in df.columns


def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def annotate(df: Union[str|pd.DataFrame], output_path: str, model_path: str, text_column: str, token: str = None) -> pd.DataFrame:
    """
    Anota a polaridade de comentários de um arquivo csv para outro arquivo csv.

    :param df: Dataframe com os dados a serem anotados
    :param output_path: onde escrever os dados anotados
    :param model_path: Caminho para o modelo do HuggingFace, armazenado em disco ou no repositório do HuggingFace.
    :param text_column: O nome da coluna que possui o comentário a ser classificado.
    :param token: Opcional - caso o modelo esteja armazenado em um repositório privado do HuggingFace, o token de
        acesso a este modelo.
    """

    model, tokenizer, pipe = __configure__(model_path, token)

    if not check_compliance(df, text_column):
        raise ValueError('A tabela de textos para ser anotados deve conter uma coluna de nome \'text\', com o '
                         'texto a ser classificado!\nVocê pode trocar o nome desta coluna após a classificação, se '
                         'assim desejar.')

    labels_values = {label: [] for label in model.config.label2id.keys()}
    for label in labels_values.keys():
        df[label] = 0.

    with tqdm(range(len(df)), desc='Anotando') as pbar:
        for i, row in df.iterrows():
            pred = pipe(row[text_column], top_k=None)

            for result in pred:
                df.loc[i, result['label']] = result['score']

            pbar.update(1)

    df.to_csv(
        output_path,
        sep=',', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False
    )
    print(f'Dataset anotado em {output_path}')
    return df


def get_input_stream(dataset_path):
    if dataset_path:
        return pd.read_csv(dataset_path, encoding='utf-8', sep=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    else:
        return pd.read_csv(sys.stdin.buffer, encoding='utf-8', sep=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


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
        '--dataset-path', action='store', required=False,
        help='Opcional - Caminho para um dataset no disco que será anotado com os sentimentos do classificador. O '
             'dataset deve possuir as seguintes características: '
             '(1) É um arquivo csv; '
             '(2) Valores separados por vírgula; '
             '(3) Valores textuais estão dentro de "aspas"; '
             '(4) Codificação do arquivo é UTF-8. '
             'Se não for fornecido, o parâmetro --dataset-file deve ser fornecido. Nesse caso, o dataset será lido da '
             'linha de comando como texto. Use esta opção para fazer piping.'
    )

    parser.add_argument(
        '--text-column', action='store', required=True,
        help='O nome da coluna que possui o comentário a ser classificado.'
    )

    args = parser.parse_args()
    dataframe = get_input_stream(args.dataset_path)

    if args.dataset_path is not None:
        output_path = args.dataset_path.replace('.csv', ' (anotado).csv')
    else:
        output_path = 'output.csv'

    if args.mode == 'annotate':
        annotate(df=dataframe, output_path=output_path, model_path=args.model_path, text_column=args.text_column, token=args.token)
    elif args.mode == 'evaluate':
        raise NotImplementedError('código defasado! fazer adaptações necessárias!')
        evaluate(df=dataframe, output_path=output_path, model_path=args.model_path, text_column=args.text_column, token=args.token)
    else:
        raise AttributeError(f'Modo de execução desconhecido: {args.mode}')
