try:
    from flask import Flask
    from flask_bootstrap import Bootstrap
except ImportError:
    raise ImportError(
        'Missing dependencies. Please run `pip install "aequitas[webapp]"`'
    )

app = Flask(__name__)
app.secret_key = "super secret key"
Bootstrap(app)
