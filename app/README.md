# Aplicação Web para análise de sentimento

## Bibliotecas usadas

* CUDA 11.8
* PyTorch 2.0.1
* transformers 4.32.1
* scikit-learn 1.3.0
* NumPy 1.24.4
* pandas 1.5.3
* Flask 2.2.2
* tqdm 4.66.1

## Uso

Após treinamento do modelo preditivo (Seção [Treino de modelos preditivos com Transformers](../learning/README.md)),
é possível carregar uma aplicação Web para testar em tempo real as predições.

Para isto, crie uma pasta `instance` dentro do diretório [app](.), e dentro desta pasta, crie um arquivo `config.py`.
Este arquivo deve ser uma cópia do arquivo [app/config.py](config.py), com os dados do caminho do modelo treinado
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