from flask import Flask
from flask_bootstrap import Bootstrap


app = Flask(__name__)
app.secret_key = 'super secret key'
Bootstrap(app)
