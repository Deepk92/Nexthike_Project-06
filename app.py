from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained Random Forest model
model_path = 'Random_forest_model_04-10-2024-11-34-37-00.pkl'
model = joblib.load(model_path)

@app.route('/', methods=['GET'])
def index():
    # Serve the HTML page for input
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request has the 'store_id' and file part
    if 'store_id' not in request.form or 'file' not in request.files:
        return jsonify({'error': 'No store_id or file part in the request'}), 400

    store_id = request.form['store_id']
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        data = pd.read_csv(file)
        # Include preprocessing as required by your model
        predictions = model.predict(data)
        return jsonify({'store_id': store_id, 'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)






