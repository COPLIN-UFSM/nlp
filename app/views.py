import flask
import os.path


def main(app: flask.app.Flask) -> flask.app.Flask:
    __script_path__ = os.path.dirname(os.path.abspath(__file__))

    @app.route('/')
    def analysis():
        return flask.render_template('analysis.html')

    @app.route('/about')
    def about():
        return flask.render_template('about.html')

    @app.errorhandler(404)
    def page_not_found(page):
        return flask.render_template('404.html'), 404

    app.register_error_handler(404, page_not_found)

    return app
