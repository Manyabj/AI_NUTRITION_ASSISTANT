from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import sys
sys.path.append("c:/Users/manya/OneDrive/Desktop/AI-ASSISTANT/backend/preprocessing")

from model_prediction import predictions  # Remove 'backend.preprocessing'



app = Flask(__name__)
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # Changed from /predictions to match server.py
def get_prediction():
    data = request.get_json()
    input_text = data.get('input', '')
    print("Received:", input_text)
    result = predictions(input_text)
    response = jsonify({'prediction': result})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True)
