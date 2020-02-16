from flask import Flask
app = Flask(__name__)

@app.route('/')
def henlo_world():
    return 'Henlo pren!'
