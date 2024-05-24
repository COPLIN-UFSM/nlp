# Anotação de comentários com a biblioteca doccano

## Bibliotecas usadas

* doccano 1.8.4

## Uso

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