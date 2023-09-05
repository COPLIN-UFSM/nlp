import flask
from flask import request, jsonify


def main(app: flask.app.Flask, classify_function, visualizer) -> flask.app.Flask:
    @app.route('/submit_text_for_analysis', methods=['POST', 'GET'])
    def submit_text_for_analysis():
        if request.method == 'POST':
            text = request.form['text_for_analysis']
        elif request.method == 'GET':
            text = request.args.get('text')
        else:
            raise ValueError(f'Método HTTP desconhecido na função submit_text_for_analysis: {request.method}')

        prediction = classify_function(text)  # pega primeiro item da lista de respostas
        pred_dict = visualizer.to_dict(visualizer.interpret_text(text))

        answer = {
            'prediction': prediction,
            'visualization': pred_dict
        }

        response = jsonify(answer)
        response.headers.add('Access-Control-Allow-Origin', '*')  # Essa linha é necessária. Requisição dos navegadores
        return response  # retorna resposta para a página Web

    return app
