# Natural Language Processing

Este repositório é uma coleção de scripts e ferramentas utilizados para tarefas de Processamento de Linguagem Natural 
(Natural Language Processing, ou NLP em inglês) realizados pela Coordenadoria de Planejamento Informacional da UFSM, 
ligada à Pró-reitoria de Planejamento - PROPLAN.

Os dados utilizados para treinar os modelos de deep learning encontram-se no repositório privado 
[nlp-data](https://github.com/COPLIN-UFSM/nlp-data). 

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

Execute o seguinte comando pela linha de comando:

```bash
conda env create -f environment.yml
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

## Uso

Este repositório compreende três aplicações distintas: 

<details>
<summary><h3>Treino de modelos preditivos com Transformers</h3></summary>

| Recurso     | Descrição                                                                                                                                                                  |
|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scripts     | [learning](learning)                                                                                                                                                       |
| Bibliotecas | <ul><li>CUDA 11.8</li><li>PyTorch 2.0.1</li><li>transformers 4.32.1</li><li>datasets 2.12.0</li><li>scikit-learn 1.3.0</li><li>NumPy 1.24.4</li><li>pandas 1.5.3</li></ul> |


Este repositório usa um modelo pré-treinado, chamado 
[BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased), e disponível no site 
[Hugging Face](https://huggingface.co/).

O modelo foi disponibilizado por Fábio Souza, Rodrigo Nogueira e Roberto Lotufo no artigo "BERTimbau: pretrained BERT 
models for Brazilian Portuguese", publicado na Brazilian Conference in Intelligent Systems (2020). Mais informações 
estão disponíveis no [repositório](https://github.com/neuralmind-ai/portuguese-bert/) do trabalho. 

Este modelo foi treinado no [BrWaC (Brazilian Web as Corpus)](https://www.researchgate.net/publication/326303825_The_brWaC_Corpus_A_New_Open_Resource_for_Brazilian_Portuguese)
para três tarefas: Reconhecimento de entidades nomeadas, similaridade textual de frases e reconhecimento de implicação 
textual. Aqui, ele passa por um ajuste-fino (fine-tuning) para classificação de sentimentos em 3 classes: positivo (o
texto em questão tem um sentimento positivo), negativo e neutro.

</details>

<details>
<summary><h3>Aplicação Web para análise de sentimento</h3></summary>

| Recurso     | Descrição                                                                                                                                                                                  |
|:------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Scripts     | [run.py](run.py), [app](app)                                                                                                                                                               |
| Bibliotecas | <ul><li>CUDA 11.8</li><li>PyTorch 2.0.1</li><li>transformers 4.32.1</li><li>scikit-learn 1.3.0</li><li>NumPy 1.24.4</li><li>pandas 1.5.3</li><li>Flask 2.2.2</li><li>tqdm 4.66.1</li></ul> |

Após treinamento do modelo preditivo (Seção [Treino de modelos preditivos com Transformers](#treino-de-modelos-preditivos-com-transformers)),
é possível carregar uma aplicação Web para testar em tempo real as predições.

Para isto, crie uma pasta `instance` dentro do diretório [app](app), e dentro desta pasta, crie um arquivo `config.py`.
Este arquivo deve ser uma cópia do arquivo [app/config.py](app/config.py), com os dados do caminho do modelo treinado
preenchidos:

```python
"""
Arquivo com definições de execução quando o backend estiver em produção
"""
# Opções de desenvolvimento: use True se estiver programando, ou False se estiver apenas testando a ferramenta
DEBUG = False  # se o debugger deve ser executado
USE_RELOADER = False  # se modificações no código-fonte devem ser atualizadas na aplicação Web
USE_CPU = False  # se o modelo deve usar a CPU para fazer classificações, ou então a GPU
PORT = 5000  # Porta a ser usada pelo servidor
MODEL_PATH = "path_to_folder_where_pytorch_model_is"  # diretório onde o modelo do PyTorch está armazenado
```

Para executar a aplicação Web, digite na linha de comando:

```bash
conda activate nlp
python run.py
```

</details>

<details>
<summary>
    <h3>
        Anotação de comentários com a biblioteca <a href="https://github.com/doccano/doccano">doccano</a>
    </h3>
</summary>

| Recurso     | Descrição                       |
|:------------|:--------------------------------|
| Scripts     | [annotation](annotation)        |
| Bibliotecas | <ul><li>doccano 1.8.4</li></ul> |

É possível utilizar a biblioteca [doccano](https://github.com/doccano/doccano) para anotar manualmente datasets para 
tarefas de Processamento de Linguagem Natural. 

Para isto, é necessário seguir o passo a passo abaixo:

1. Instalar a biblioteca de anotação:

   ```bash
   conda activate nlp
   pip install doccano
   ```

2. Realizar a configuração inicial:

   ```bash
   doccano init  # inicializa base de dados
   doccano createuser --username admin --password pass # cria um super-usuário
   ```

3. A ferramenta necessita de dois processos executando ao mesmo tempo para funcionar, ambos iniciados pela linha de 
comando. É possível executá-los de diversas maneiras: 
   * Abrir duas janelas do terminal; 
   * Instalar a ferramenta tmux no Linux (`apt-get install tmux`);
   * Executar o script [doccano.sh](annotation/doccano.sh) (para Linux) a partir da linha de comando.

   Os processos são:

   **Ferramenta de anotação**

   ```bash
   doccano webserver --port 8000
   ```
   
   **Uploader de arquivos**

   ```bash
   doccano task
   ```

4. Para criar ou remover usuários, acesse a url `http://localhost:8000/admin/auth` ou `http://localhost:8000/admin/`, 
   sendo `localhost` a URL onde o serviço está hospedado (localhost se for a própria máquina) e 8000 a porta onde o 
   serviço está disponibilizado.

</details>

## Contato

Desenvolvido originalmente por Henry Cagnini [henry.cagnini@ufsm.br]() e Raphael Amaro [raphael.amaro@ufsm.br]().

## Bibliografia

* [PyTorch Tutorials](https://pytorch.org/tutorials/)
