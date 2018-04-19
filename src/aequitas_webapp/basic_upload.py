import itertools
import os.path
import tempfile

import pandas as pd
from flask import (
    abort,
    Markup,
    request,
    redirect,
    url_for,
    flash,
    render_template,
)
from werkzeug.utils import secure_filename

from aequitas.fairness import Fairness
from aequitas.preprocessing import preprocess_input_df

from aequitas_cli.aequitas_audit import audit
from aequitas_cli.utils.configs_loader import Configs

from . import app


HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_DATA = {
    'sample1': os.path.join(HERE, 'sample_data/compas_for_aequitas.csv'),
    'sample2': os.path.join(HERE, 'sample_data/adult_rf_binary.csv'),
}


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file_ = request.files.get('file')

        if not file_ or not file_.filename:
            flash('No selected file')
            return redirect(request.url)

        (name, ext) = os.path.splitext(file_.filename)
        if not ext.lower() == '.csv':
            flash('Bad file type')
            return redirect(request.url)

        dirpath = tempfile.mkdtemp(prefix='')
        filename = secure_filename(file_.filename)
        file_.save(os.path.join(dirpath, filename))
        return redirect(url_for("uploaded_file",
                                dirname=os.path.basename(dirpath),
                                filename=name))

    return render_template("file_upload.html")


@app.route('/about.html', methods=['GET'])
def about():
    return render_template("about.html")


@app.route('/audit/<dirname>/<name>/', methods=['GET', 'POST'])
@app.route('/audit/<name>/', methods=['GET'])
def uploaded_file(name, dirname='sample'):
    if dirname == 'sample':
        if name not in SAMPLE_DATA:
            abort(404)

        source_path = SAMPLE_DATA[name]
        filename = os.path.basename(source_path)
        (name, _ext) = os.path.splitext(filename)
        dirpath = tempfile.mkdtemp(prefix='')
        os.symlink(source_path, os.path.join(dirpath, filename))

        return redirect(url_for('uploaded_file',
                                dirname=os.path.basename(dirpath),
                                name=name))

    upload_path = os.path.join(tempfile.gettempdir(), dirname)
    data_path = os.path.join(upload_path, name + '.csv')
    if not os.path.exists(data_path):
        abort(404)

    df = pd.read_csv(data_path)
    (df, groups) = preprocess_input_df(df)
    subgroups = {col: list(set(df[col])) for col in groups}

    if "submit" not in request.form:
        return render_template("customize_report2.html",
                               categories=groups,
                               subcategories=subgroups,
                               fairness=('Equal Parity',
                                         'Proportional Parity',
                                         'False Positive Parity',
                                         'False Negative Parity'))

    rgm = request.form["ref_groups_method"]
    if rgm == 'predefined':
        group_variables = request.form.getlist('group_variable1')
    else:
        group_variables = request.form.getlist('group_variable2')

    # check if user forgot to select anything; return all
    if len(group_variables) == 0:
        group_variables = groups

    # remove unwanted cols from df
    subgroups = {g: request.form[g] for g in group_variables}

    # majority_groups = request.form.getlist('use_majority_group')
    raw_fairness_measures = request.form.getlist('fairness_measures')
    if len(raw_fairness_measures) == 0:
        fairness_measures = list(Fairness().fair_measures_supported)
    else:
        # map selected measures to input
        fair_map = {'Equal Parity': ['Statistical Parity'],
                    'Proportional Parity': ['Impact Parity'],
                    'False Positive Parity': ['FPR Parity', 'FDR Parity'],
                    'False Negative Parity': ['FNR Parity', 'FOR Parity']
                    }
        fairness_measures = [y for x in raw_fairness_measures
                             for y in fair_map[x]]

        fairness_pct = request.form['fairness_pct']
        try:
            fp = 1.0 - float(fairness_pct) / 100.0
        except:
            fp = 0.8

        configs = Configs(ref_groups=subgroups,
                          ref_groups_method=rgm,
                          fairness_threshold=fp,
                          fairness_measures=fairness_measures,
                          attr_cols=group_variables)

        (_gv_df, report) = audit(df,
                                 model_id=1,
                                 configs=configs,
                                 preprocessed=True)

        for reportid in itertools.count(1):
            report_path = os.path.join(upload_path, str(reportid))
            if not os.path.exists(report_path):
                break

        with open(report_path, 'w') as fd:
            fd.write(report)

        return redirect(url_for("report",
                                dirname=dirname,
                                name=name,
                                reportid=reportid))


@app.route('/audit/<dirname>/<name>/report-<reportid>.html', methods=['GET'])
def report(dirname, name, reportid):
    report_path = os.path.join(tempfile.gettempdir(), dirname, reportid)
    if not os.path.exists(report_path):
        abort(404)

    with open(report_path) as fd:
        report = fd.read()

    return render_template(
        'report.html',
        content=Markup(report),
        go_back=url_for('uploaded_file', dirname=dirname, name=name),
    )
