#!/usr/bin/env python
import importlib

# Make WSGI app available to server as "application"
from aequitas_webapp import app as application


# Ensure webapp logic is loaded
# (but don't import it into this namespace)
importlib.import_module('aequitas_webapp.views')


if __name__ == '__main__':
    # Run development server
    application.run()
