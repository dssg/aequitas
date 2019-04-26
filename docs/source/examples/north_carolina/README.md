# ncdoc_data
North Carolina's Department of Public Safety posts "[all public information on all NC Department of Public Safety offenders convicted since 1972](http://webapps6.doc.state.nc.us/opi/downloads.do?method=view)." Inspired by a previous scrape by Fred Whitehurst, Isabella Langan, and Tom Workman, I wrote scripts to download the data and save in CSV format.

To get the data, run `./ncdoc_parallel.sh`. It will download and transform the data and store the outputs in the `preprocessed/` directory.

Requirements:
- [bash](https://www.gnu.org/software/bash/)
- [csvkit](https://github.com/wireservice/csvkit)
- [GNU parallel](https://www.gnu.org/software/parallel/)
- [Python](https://www.python.org/downloads/)
- [Numpy](https://docs.scipy.org/doc/numpy-1.15.0/user/install.html)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
- [Jupyter](https://jupyter.org/install)

Files:
- `ncdoc_des2csv.sh` - downloads and processes one zip file into a CSV
- `ncdoc_parallel.sh` - runs `ncdoc_des2csv.sh` in parallel for all the necessary files
- `fixed_width_definitions_format.csv` - gives necessary data for unzipping files into CSVs
- `create_recidivism_set.ipynb` - Jupyter notebook which processes the raw data into one large dataset for predicting recidivism
- `create_matrices.ipynb` - Jupyter notebook which takes the resulting dataset from `create_recidivism_set.ipynb` and breaks it into a series of paired training and testing matrices resembling the output format of [Triage](https://github.com/dssg/triage)
