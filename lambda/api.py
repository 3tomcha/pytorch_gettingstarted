from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/cats")
def cats():
  return "Cats"

@app.route("/dogs/<id>")
def dog(id):
  return "Dog"
