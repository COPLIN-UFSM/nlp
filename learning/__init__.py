import torch
from transformers import BertForSequenceClassification, BertTokenizer, pipeline


def load_model(model_path: str, use_cpu: bool = False):
    """
    Carrega um Pipeline para análise de sentimento a partir de um caminho para um modelo.

    :param model_path: Caminho para a pasta de um modelo de predição.
    :param use_cpu: Opcional - se o modelo deve rodar na CPU. Ative para computadores sem uma GPU da NVIDIA compatível.
    :return: Um Pipeline, pronto para classificação.
    """
    device = torch.device('cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Using {device} as hardware')

    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lowercase=False)
    pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, top_k=None)

    return pipe
