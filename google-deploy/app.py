from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/pridict', methods=['POST'])
def predict():
    pass

