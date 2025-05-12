from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Looks in 'templates/index.html'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_val = data.get('Open')
    input_df = pd.DataFrame({'Open': [input_val]})
    prediction = model.predict(input_df)
    return jsonify({'predicted_close': prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
