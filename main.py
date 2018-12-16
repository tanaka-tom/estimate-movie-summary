from flask import Flask, request, jsonify
from word2vec_predict import SearchSimilarWords

app = Flask(__name__)


@app.before_first_request
def _load_model():
    global model

    model = SearchSimilarWords()


@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.route('/estimate')
def estimate():
    if not model:
        _load_model()
        if not model:
            return 'Model not found.'

    movie_id = int(request.args.get('id'))

    result = model.cal(movie_id)
    return jsonify({
        'status': 'OK',
        'result': result
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0')
