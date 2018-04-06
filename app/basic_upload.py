import os
import sys
from flask import Flask, request, redirect, url_for, send_from_directory, flash, render_template, session
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append(os.path.abspath(os.path.join('../bin')))

from src.aequitas.group import Group
from src.aequitas.bias import Bias
from src.aequitas.fairness import Fairness
from src.aequitas.preprocessing import preprocess_input_df

from bin.aequitas_audit import audit
from bin.utils.configs_loader import Configs

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

app = Flask(__name__)
app.secret_key = 'super secret key'
Bootstrap(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            df = pd.read_csv(request.files.get('file'))
            session['df'] = df.to_json()
            return redirect(url_for("uploaded_file"))
    return render_template("file_upload.html")


@app.route('/customize', methods=['get', 'post'])
def uploaded_file():
    necessary_cols = ['model_id', 'entity_id', 'score', 'label_value', 'rank_abs', 'rank_pct']
    #df, groups = preprocess_input_df(pd.read_csv(app.config['UPLOAD_FOLDER']+'/'+uuid))
    df, groups = preprocess_input_df(pd.read_json(session['df']))
    subgroups = {}
    for col in groups:
        subgroups[col] = (list(set(df[col])))

    if "submit" not in request.form:
        fairness_measures = ['Statistical Parity',
                            'Impact Parity',
                            'False Positive Rate Parity',
                            'False Omission Rate Parity',
                            'False Negative Rate Parity',
                            'False Discovery Rate Parity']
        return render_template("customize_report2.html",
                                           categories=groups,
                                           subcategories = subgroups,
                                           fairness = fairness_measures)
    else:
        print(request.form)
        group_variables = request.form.getlist('group_variable')
        # check if user forgot to select anything; return all
        if len(group_variables)==0:
            group_variables = groups
        # remove unwanted cols from df
        df = df[necessary_cols+group_variables]
        subgroups = {g:request.form[g] for g in group_variables}
        majority_groups = request.form.getlist('use_majority_group')
        fairness_measures = request.form.getlist('fairness_measures')
        fairness_pct = request.form['fairness_pct']
        rgm = request.form["ref_groups_method"]

        try:
            fp = 100-int(fairness_pct)
        except:
            fp = 80

        configs = Configs(ref_groups=subgroups,
                          ref_groups_method=rgm,
                          report=False,
                          fairness_threshold=fp)

        gv_df = audit(df, model_id=1, configs=configs, preprocessed=True)

        return gv_df.to_html()

    '''
    <!doctype html>
    <title>Good job</title>
    <h1>u did it</h1>
    '''

if __name__ == "__main__":
    app.run()