# Using nbsphinx to make docs


Make changes in python notebooks stored in `/source` 

```
python3 -m pip install nbsphinx
cd aequitas
python3 -m sphinx docs/source docs
```


To see changes locally in a live webbrowser use sphinx-autobuild

```
python3 -m pip install sphinx-autobuild --user
python3 -m sphinx_autobuild docs/source docs
```

To add a new file: 
- Create a notebook (or rst file) in `/source` 
- Add file name to toctree in `/source/index.rst`

For more advanced features, here are the [nbshinx docs](http://nbsphinx.readthedocs.io/en/0.3.3/)

