import os
import sys
import socket

from flask import Flask

from app.views import main as views_func
from app.models import main as models_func

from learning.interpret import Visualizer
from learning.singlelabel.predict import classify, load_model


def get_host():
    host = 'localhost'

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))
        host = s.getsockname()[0]

    return host


def main():
    # configura a aplicação e define as pastas onde ela deve procurar os itens
    current_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1])

    app = Flask(
        'Classificador de Sentimentos COPLIN-UFSM',
        template_folder='templates',
        static_folder='static',
        root_path=os.path.join(current_path, 'app'),
        instance_path=os.path.join(current_path, 'app', 'instance'),
        instance_relative_config=True
    )
    sys.path.append(os.path.join(current_path, 'app'))
    # configura os arquivos de definições, app/config.py e app/instance/config.py
    app.config.from_object('config')
    try:
        app.config.from_pyfile('config.py')
    except FileNotFoundError:
        print('--' * 55, file=sys.stderr)
        print(
            ' O arquivo app/instance/config.py não foi encontrado, '
            'então o arquivo app/config.py será usado no seu lugar.',
            file=sys.stderr
        )
        print('--' * 55, file=sys.stderr)

    # parte de classificação de modelos
    model_path = app.config["MODEL_PATH"]
    print(f'Usando modelo de {model_path}')

    pipe = load_model(model_path, use_cpu=app.config['USE_CPU'])

    visualizer = Visualizer(pipe, max_length=128)

    # carrega as definições de funções ajax na aplicação
    app = models_func(app, lambda x: classify(pipe, x), visualizer)
    app = views_func(app)  # carrega as definições de roteamento na aplicação

    # se debug=True, coloca o backend a rodar no modo debug; modificações feitas nos arquivos de código-fonte
    # se refletirão em tempo real nas páginas Web (basta dar um F5 no navegador)
    app.run(
        host=get_host(),
        port=app.config["PORT"],
        debug=app.config['DEBUG'],
        use_reloader=app.config["USE_RELOADER"]
    )


if __name__ == '__main__':
    main()
