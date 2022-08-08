from flask import Flask, jsonify, request
from generate_text import get_aggregated_completions
from offensive_classifier import sort_offensive
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST'])
def predictions():
    """Sorts predictions by offensiveness for redteaming"""
    req = request.get_json()
    generated = get_aggregated_completions(req['prompt'], req['numPredictions'])
    sorted_greedy = sort_offensive(generated['greedy'])
    sorted_beam = sort_offensive(generated['beam'])
    return_dict = {
        'attention': generated['attention'].tolist(),
        'beam': sorted_beam,
        'greedy': sorted_greedy,
        'tokens': [token.replace('Ġ', '') for token in generated['tokens']]}
    return jsonify(return_dict)

if __name__ == '__main__':
    app.run()
