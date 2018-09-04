#!/usr/bin/env python
import importlib
import os
# Make WSGI app available to server as "application"
from aequitas_webapp import app as application


# Ensure webapp logic is loaded
# (but don't import it into this namespace)
importlib.import_module('aequitas_webapp.views')

if __name__ == '__main__':
    host = os.environ.get('HOST', '127.0.0.1')
    # Run development server
    application.run(host=host)
