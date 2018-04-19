import importlib

from . import app as application


importlib.import_module('aequitas_webapp.views')


if __name__ == '__main__':
    application.run()
