# Natural Language Processing

Este repositório é uma coleção de scripts e ferramentas utilizados para tarefas de Processamento de Linguagem Natural
(Natural Language Processing, ou NLP em inglês) realizados pela Coordenadoria de Planejamento Informacional da UFSM,
ligada à Pró-reitoria de Planejamento - PROPLAN.

Os dados utilizados para treinar os modelos de deep learning encontram-se no repositório privado
[nlp-data](https://github.com/COPLIN-UFSM/nlp-data).

Este repositório compreende três aplicações distintas:

* [Treino de modelos preditivos com Transformers](learning/sentiment_analysis/README.md)
* [Aplicação Web para análise de sentimento](app/README.md)
* [Anotação de comentários com a biblioteca doccano](DOCCANO.md)

## Sumário

* [Pré-requisitos](#pré-requisitos)
* [Instalação](#instalação)
* [Instruções de uso](#uso)
* [Contato](#contato)
* [Bibliografia](#bibliografia)

## Pré-requisitos

Este repositório requer a última versão do [Python Anaconda](https://www.anaconda.com/download) para ser executado,
visto que usa o gerenciador de pacotes conda. O código executará em qualquer Sistema Operacional, mas foi desenvolvido
originalmente para Windows 10 Pro e Ubuntu 22.04.3 LTS (ambos 64 bits).

Também é necessário instalar a versão compatível das bibliotecas [CUDA](https://developer.nvidia.com/cuda-downloads) e
[PyTorch](https://pytorch.org/get-started/locally/#anaconda). Clique em cada um dos links anteriores e siga os tutoriais
para baixar a versão adequada para a sua máquina.

As configurações da máquina que o repositório foi desenvolvido encontram-se na tabela abaixo:

| Configuração        | Valor                              |
|---------------------|------------------------------------|
| Sistema operacional | Windows 10 Pro /Ubuntu 22.04.3 LTS |
| Processador         | Intel core i7 9700                 |
| Memória RAM         | 16GB                               |
| Placa de vídeo      | Nvidia GTX 730                     |
| Memória de vídeo    | 2GB                                |
| Versão do CUDA      | 11.8                               |
| Necessita rede?     | Não                                |

## Instalação

> [!WARNING]
> Infelizmente, não é possível usar um arquivo `environment.yml` para configuração do ambiente virtual. 
>

Para criar o ambiente virtual com as bibliotecas para execução na GPU, execute os seguintes comandos, nesta ordem:

```bash
conda create --name nlp python==3.11.* pip --yes  
conda activate nlp
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
conda install captum -c pytorch --yes
conda install --file requirements.txt --yes
pip install --requirement pip_requirements.txt
```

Para utilizar aceleração por GPU no treinamento dos algoritmos de deep learning (não necessário para execução de modelos
já treinados), execute o seguinte passo a passo:

```bash
conda activate nlp
python
```

E então, dentro do console Python:

```python
import torch
torch.cuda.is_available()
```

A resposta deve ser `True`, caso uma placa de vídeo NVIDIA compatível esteja disponível. A disponibilidade depende dos
drivers mais recentes estarem instalados.

## Instruções de uso

> [!NOTE]
> Para classificação dos comentários da avaliação de ensino-aprendizagem, siga para [Análise de sentimento](learning/sentiment_analysis/README.md).

## Contato

Desenvolvido originalmente por Henry Cagnini [henry.cagnini@ufsm.br]() e idealizado por Raphael Amaro [raphael.amaro@ufsm.br]().

## Bibliografia

* [PyTorch Tutorials](https://pytorch.org/tutorials/)
