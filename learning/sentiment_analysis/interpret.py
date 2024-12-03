__description__ = """
    Código fonte adaptado de
    https://towardsdatascience.com/interpreting-the-prediction-of-bert-model-for-text-classification-5ab09f8ef074
    
    Script para identificar quais palavras em uma sentença colaboram para a predição do sentimento.
"""

import argparse
import os

import captum
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients, visualization
from captum.attr._utils.visualization import VisualizationDataRecord
from lxml import etree, html
from transformers import BertTokenizer, Pipeline

from learning import load_model


class Visualizer(object):
    def __init__(self, pipe, max_length=128):
        self.pipe = pipe

        self.max_length = max_length

        self.model_input = self.pipe.model.bert.embeddings
        self.model_output = lambda input: self.pipe.model(input)[0]

        self.lig = LayerIntegratedGradients(self.model_output, self.model_input)  # type: LayerIntegratedGradients

    @staticmethod
    def __summarize_attributions__(attributions):
        # squeeze(0) = remove uma dimensão
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        return attributions

    def interpret_text(self, text: str, true_label: str = None):
        """
        Analisa uma predição feita por uma rede neural profunda para a tarefa de análise de sentimento.

        :param text: texto a ser analisado.
        :param pipe: pipeline que contém o modelo e o tokenizer usado pelo modelo
        :param true_label: Opcional - Rótulo original do dado apresentado.
        """

        target_pred = self.pipe(text)[0][0]
        target_label = target_pred['label']
        target_prob = target_pred['score']

        input_ids, baseline_input_ids, text_tokens = self.construct_input_and_baseline(text)

        attributions, delta = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline_input_ids,
            return_convergence_delta=True,
            target=self.pipe.model.config.label2id[target_label]
        )
        # soma das atribuições das palavras
        word_attributions = self.__summarize_attributions__(attributions)

        output = self.pipe.model(input_ids)[0]
        score_vis = visualization.VisualizationDataRecord(
            word_attributions=word_attributions,  # a atenção que cada palavra dá para aquela classe
            pred_prob=target_prob,
            pred_class=target_label,
            true_class=true_label,
            attr_class=text,
            attr_score=word_attributions.sum(),  # a soma da atenção da frase
            raw_input_ids=text_tokens,
            convergence_score=delta
        )

        return score_vis

    def construct_input_and_baseline(self, text):
        """
        Transforma um texto num conjunto de tokens, definidos pelo tokenizer.

        :param text: texto de entrada
        :return: Uma tupla onde:
            * Primeiro item é um tensor com lista de tokens do texto de entrada;
            * Segundo item é um tensor com uma lista de tokens em branco (baseline_tokens);
            * A lista de tokens de entrada, como uma lista Python
        """
        tokenizer = self.pipe.tokenizer  # type: BertTokenizer
        device = self.pipe.device  # type: torch.device

        text_ids = tokenizer.encode(text, max_length=self.max_length, truncation=True, add_special_tokens=True)

        baseline_input_ids = [tokenizer.pad_token_id] * len(text_ids)

        text_tokens = tokenizer.convert_ids_to_tokens(text_ids)

        return torch.tensor([text_ids], device=device), torch.tensor([baseline_input_ids], device=device), text_tokens

    @staticmethod
    def to_dict(register: VisualizationDataRecord):
        word_attention = [x.item() for x in register.word_attributions][1:-1]
        text_tokens = register.raw_input_ids[1:-1]

        data = {
            'text': register.attr_class,
            'text_tokens': text_tokens,
            'text_attention': register.attr_score.item(),
            'word_attention': word_attention,
            'convergence_score': register.convergence_score.item(),
            'pred_class': register.pred_class,
            'pred_prob': register.pred_prob
        }
        return data

    @staticmethod
    def write_html_to_file(visualizations: list, path_write_html: str):
        """
        Gera um arquivo HTML com as visualizações feitas.

        :param visualizations: Lista de visualizações geradas.
        :param path_write_html: Caminho para escrever o arquivo HTML.
        """
        html_ipython = visualization.visualize_text(visualizations)

        document_root = html.fromstring(html_ipython.data)

        print(f'Visualization image written to {path_write_html}')

        with open(path_write_html, 'wb') as file:
            file.write(bytes('<meta charset="utf-8">\n', 'utf-8'))
            file.write(etree.tostring(document_root, encoding='utf-8', pretty_print=True, method='html'))

    @staticmethod
    def print_libraries():
        print('using following libraries:')
        versions = pd.DataFrame(
            [(package.__name__, package.__version__) for package in [captum, torch]],
            columns=['package', 'version']
        )
        print(versions)


def main(model_path: str, instances: list, use_cpu: bool = False, write: bool = False):
    pipe = load_model(model_path, use_cpu=use_cpu)  # type: Pipeline

    vis = Visualizer(pipe, max_length=128)
    vis.print_libraries()

    visualizations = []
    dict_vis = []
    for text, true_label in instances:
        register = vis.interpret_text(text, true_label)
        visualizations += [register]
        dict_vis += [vis.to_dict(register)]

    if write:
        vis.write_html_to_file(visualizations, os.path.join(model_path, 'visualization.html'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script para aplicar a técnica de Integrated Gradients em um modelo treinado.'
    )

    parser.add_argument(
        '--model-path', action='store', required=True,
        help='Caminho para uma pasta com o modelo treinado.'
    )

    parser.add_argument(
        '--write', action='store_true', required=False, default=False,
        help='Se um arquivo com as visualizações deve ser escrito em disco.'
    )

    parser.add_argument(
        '--use-cpu', action='store_true', required=False, default=False,
        help='Opção para usar CPU ao invés de GPU. Use-a caso sua placa de vídeo não seja compatível com a versão do '
             'pyTorch usada (>= 1.7).'
    )

    args = parser.parse_args()

    _instances = [
        ('eu gosto muito do professor fulano! é o melhor professor do mundo!', 'positive'),
        ('este professor é horrível!', 'negative'),
        ('é o melhor professor do universo!', 'positive')
    ]

    main(model_path=args.model_path, instances=_instances, use_cpu=args.use_cpu, write=args.write)
