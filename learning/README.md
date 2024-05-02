# Treino de modelos preditivos com Transformers

## Bibliotecas usadas

* CUDA 12.4
* PyTorch 2.3
* PyTorch-CUDA 11.8
* Transformers 4.37.2
* datasets 2.12.0
* scikit-learn 1.4.2
* NumPy 1.24.3
* pandas 1.5.3

Este repositório usa um modelo pré-treinado, chamado [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased), e disponível no site [Hugging Face](https://huggingface.co/).

O modelo foi disponibilizado por Fábio Souza, Rodrigo Nogueira e Roberto Lotufo no artigo "BERTimbau: pretrained BERT
models for Brazilian Portuguese", publicado na Brazilian Conference in Intelligent Systems (2020). Mais informações
estão disponíveis no [repositório](https://github.com/neuralmind-ai/portuguese-bert/) do trabalho.

Este modelo foi treinado no [BrWaC (Brazilian Web as Corpus)](https://www.researchgate.net/publication/326303825_The_brWaC_Corpus_A_New_Open_Resource_for_Brazilian_Portuguese)
para três tarefas: Reconhecimento de entidades nomeadas, similaridade textual de frases e reconhecimento de implicação
textual. Aqui, ele passa por um ajuste-fino (fine-tuning) para classificação de sentimentos em 3 classes: positivo (o
texto em questão tem um sentimento positivo), negativo e neutro.

## Uso

Para usar este script:

1. A partir do diretório raiz do repositório, crie uma pasta `instance´. Dentro dela, crie uma pasta `models`. 
  Finalmente, dentro da pasta `models`, crie outra pasta, desta vez com o nome do modelo que será treinado. Neste 
  exemplo usaremos o nome `multilabel_two_classes`, mas você pode usar qualquer outro nome.

   ```
   nlp/
     instance/
       models/
         multilabel_two_classes/
   ```

2. Copie-e-cole para dentro da pasta do modelo o arquivo [parameters.json](../parameters.json):

   ```
   nlp/
     instance/
       models/
         multilabel_two_classes/
           parameters.json
   ```

3. Abra este arquivo em um editor de texto, e mude os parâmetros de acordo com sua preferência. Os principais parâmetros
   a serem modificados são:

   * use_cpu: Use `true` caso você não tenha uma placa de vídeo NVIDIA compatível com CUDA. Para saber se sua placa é 
     compatível, execute o comando `python -c "import torch; print(torch.cuda.is_available())`; caso a saída deste 
     comando seja `True`, você tem uma placa de vídeo compatível e configurada.
   * num_train_epochs: número de épocas para treinar o modelo. Para fazer um fine-tuning, não é necessário utilizar 
     muitas épocas. Modifique de acordo com sua preferência;
   * train_path, val_path, test_path: caminhos para arquivos de treino, validação e teste, respectivamente. Os arquivos 
     devem estar no formato csv, possuírem delimitação por vírgula, texto entre aspas, e codificação UTF-8. Use caminho
     absoluto.
   * input_column: nome da coluna nos arquivos de treino, validação e teste que possuí o texto.
   * class_name: nome das colunas que serão utilizadas como atributo-classe, uma coluna para cada rótulo. Estas colunas
     devem ser binárias (i.e. valor 1 para um comentário que apresenta aquele sentimento, ou 0 em caso contrário).
   * output_dir: caminho onde escrever o modelo treinado. Use um caminho absoluto para o diretório 
     `instance/models/<nome_do_modelo>`, e.g. 
     `C:\\Users\\henry\\Projects\\nlp\\instance\\models\\multilabel_two_classes` 
   * output_model_name: Nome do modelo. Neste tutorial, estamos usando `multilabel_two_classes`

4. Após a configuração, e a partir da pasta raiz do diretório, execute o script [finetune.py](multilabel/finetune.py), 
   passando como parâmetro o caminho do arquivo `parameters.json`:

   ```bash
   conda activate nlp
   python learning/multilabel/finetune.py --parameters-path instance/models/<nome_do_modelo>/parameters.json 
   ```
  
   Substitua `<nome_do_modelo>` pelo nome dado para o parâmetro `output_model_name`.