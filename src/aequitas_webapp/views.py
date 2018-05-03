import itertools
import os.path
import tempfile

import pandas as pd
from aequitas.fairness import Fairness
from aequitas.preprocessing import preprocess_input_df
from aequitas_cli.aequitas_audit import audit
from aequitas_cli.utils.configs_loader import Configs
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

from . import app

HERE = os.path.dirname(os.path.abspath(__file__))

SAMPLE_DATA = {
    'sample1': os.path.join(HERE, 'sample_data/compas_for_aequitas.csv'),
    'sample2': os.path.join(HERE, 'sample_data/adult_rf_binary.csv'),
}


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


# FIXME: This is not used
@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/audit/', methods=['POST'])
def upload_file():
    referer = request.headers.get('referer')
    redirect_url = referer or url_for('home')

    file_ = request.files.get('file')

    if not file_ or not file_.filename:
        flash('Please select a file', 'warning')
        return redirect(redirect_url)

    (name, ext) = os.path.splitext(file_.filename)
    if not ext.lower() == '.csv':
        flash('Bad file type – CSV required', 'warning')
        return redirect(redirect_url)

    dirpath = tempfile.mkdtemp(prefix='')
    filename = secure_filename(file_.filename)
    file_.save(os.path.join(dirpath, filename))
    return redirect(url_for('audit_file',
                            dirname=os.path.basename(dirpath),
                            name=name))


@app.route('/audit/<name>/', methods=['GET'])
def audit_sample(name):
    if name not in SAMPLE_DATA:
        abort(404)

    source_path = SAMPLE_DATA[name]
    filename = os.path.basename(source_path)
    (name, _ext) = os.path.splitext(filename)
    dirpath = tempfile.mkdtemp(prefix='')
    os.symlink(source_path, os.path.join(dirpath, filename))
    return redirect(url_for('audit_file',
                            dirname=os.path.basename(dirpath),
                            name=name))


FAIR_MAP = {'Equal Parity': ['Statistical Parity'],
            'Proportional Parity': ['Impact Parity'],
            'False Positive Parity': ['FPR Parity', 'FDR Parity'],
            'False Negative Parity': ['FNR Parity', 'FOR Parity']}

REVERSE_FAIR_MAP = {'Statistical Parity': 'Equal Parity',
                    'Impact Parity': 'Proportional Parity',
                    'FPR Parity': 'False Positive Parity',
                    'FDR Parity': 'False Positive Parity',
                    'FNR Parity': 'False Negative Parity',
                    'FOR Parity': 'False Negative Parity'}


@app.route('/audit/<dirname>/<name>/', methods=['GET', 'POST'])
def audit_file(name, dirname):
    upload_path = os.path.join(tempfile.gettempdir(), dirname)
    data_path = os.path.join(upload_path, name + '.csv')
    if not os.path.exists(data_path):
        abort(404)

    try:
        df = pd.read_csv(data_path)
    except pd.errors.ParserError:
        flash('Bad CSV file – could not parse', 'warning')
        return redirect(url_for('home'))

    (df, groups) = preprocess_input_df(df)

    if "submit" not in request.form:
        subgroups = {col: list(set(df[col])) for col in groups}

        # set defaults
        for (key, values) in (
            ('race', ('White', 'Caucasian')),
            ('sex', ('Male',)),
            ('gender', ('Male',)),
            ('age_cat', ('25 - 45',)),
            ('education', ('HS-grad',)),
        ):
            if key in subgroups:
                subgroups[key].sort(key=lambda value: int(value not in values))

        supported_fairness_measures = Fairness().get_fairness_measures_supported(df)
        fairness_measures = {REVERSE_FAIR_MAP[x] for x in supported_fairness_measures}

        return render_template('audit.html',
                               categories=groups,
                               subcategories=subgroups,
                               fairness=fairness_measures)

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
        fairness_measures = list(Fairness().get_fairness_measures_supported(df))
    else:
        # map selected measures to input
        fairness_measures = [y for x in raw_fairness_measures for y in FAIR_MAP[x]]

    try:
        fv = float(request.form['fairness_pct'])
    except (KeyError, ValueError):
        fp = 0.8
    else:
        fp = fv / 100.0
    if fv == 0:
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
        go_back=url_for('audit_file', dirname=dirname, name=name),
    )
