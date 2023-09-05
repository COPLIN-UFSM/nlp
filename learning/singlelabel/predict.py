import json
import argparse

import datasets
import pandas as pd
from datasets import Dataset
from evaluate import evaluator
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Pipeline

from learning import load_model
from learning.singlelabel.finetune import load_single_label_dataset


def classify(pipe: Pipeline, text: str) -> list:
    """
    Classifica um texto usando um modelo de análise de sentimento.

    :param pipe: Um Pipeline para classificação de texto.
    :param text: O texto a ser classificado.
    :return: Uma lista de dicionários. Cada dicionário possui duas informações: label (o rótulo associado) e score
        (a confiança do modelo na predição)
    """
    predict = pipe(text)[0]
    return predict


def evaluate(model_path, parameters_path):
    with open(parameters_path, 'r') as parameters_file:
        parameters = json.load(parameters_file)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=parameters['max_length'])

    df, test_set, num_labels, labels, label2id, id2label = load_single_label_dataset(
        parameters['test_path'], parameters, tokenizer
    )

    task_evaluator = evaluator('sentiment-analysis')

    test_dataset = Dataset.from_pandas(df)
    test_dataset = test_dataset.cast_column('label', datasets.ClassLabel(names=labels))

    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        data=test_dataset,
        tokenizer=tokenizer,
        label_mapping=label2id,
        device=-1 if parameters['use_cpu'] else 0,  # TODO change number later to match available gpu
        input_column='text',
        label_column='label'
    )
    print(pd.DataFrame.from_dict(eval_results, orient='index'))


def main(model_path: str):
    pipe = load_model(model_path)

    text = 'eu gosto muito do professor henry! é o melhor professor do mundo'

    predict = pipe(text)

    df = pd.DataFrame(predict[0]).sort_values(by='score', ascending=False)

    print(f'texto: {text}\n{df}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script que faz a classificação de sentimento de textos, dado um caminho para um modelo.'
    )

    parser.add_argument(
        '--model-path', action='store', required=True,
        help='Caminho para uma pasta onde o modelo treinado está armazenado.'
    )

    parser.add_argument(
        '--parameters-path', action='store', required=True,
        help='Caminho para um arquivo com os parâmetros de treinamento, bem como caminho dos datasets.'
    )

    args = parser.parse_args()
    main(model_path=args.model_path)
    # evaluate(model_path=args.model_path, parameters_path=args.parameters_path)

