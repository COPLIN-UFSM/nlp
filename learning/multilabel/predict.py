import csv
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from transformers import pipeline, Pipeline, BertTokenizer, BertForSequenceClassification


__description__ = 'Script que faz a classificação multi-rótulo e multi-classe de sentimento em textos.'

from learning import load_model


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


def __common__(model_path: str, dataset_path: str) -> tuple:
    model = BertForSequenceClassification.from_pretrained(model_path)  # type: BertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained(model_path, model_max_length=model.config.max_position_embeddings)
    pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    df = pd.read_csv(dataset_path, sep=',', quotechar='"', encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    return model, tokenizer, pipe, df


def evaluate(model_path: str, dataset_path: str) -> dict:
    model, tokenizer, pipe, df = __common__(model_path, dataset_path)

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

def annotate(model_path: str, dataset_path: str) -> pd.DataFrame:
    model, tokenizer, pipe, df = __common__(model_path, dataset_path)

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
    print(f'Dataset anotado escrito em {new_path}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__description__
    )

    parser.add_argument(
        '--mode', action='store', required=True,
        help='Modo que este script será rodado: \'annotate\', \'evaluate\''
    )

    parser.add_argument(
        '--model-path', action='store', required=True,
        help='Um dos dois: caminho para uma pasta onde o modelo treinado está armazenado, OU URL do repositório '
             'HuggingFace'
    )

    parser.add_argument(
        '--dataset-path', action='store', required=False,
        help='Opcional - caminho para um dataset no disco que será anotado com os sentimentos do classificador. O '
             'dataset deve possuir as seguintes características:\n'
             '* É um arquivo csv;'
             '* Valores separados por vírgula;'
             '* Valores textuais estão dentro de "aspas";'
             '* Codificação do arquivo é UTF-8.'
    )

    args = parser.parse_args()

    if args.mode == 'annotate':
        annotate(model_path=args.model_path, dataset_path=args.dataset_path)
    elif args.mode == 'evaluate':
        evaluate(model_path=args.model_path, dataset_path=args.dataset_path)
    else:
        raise AttributeError(f'Modo de execução desconhecido: {args.mode}')
