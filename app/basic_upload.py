import os
import sys
from flask import Flask, request, redirect, url_for, send_from_directory, flash, render_template
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

from bin.aequitas_audit import audit


UPLOAD_FOLDER = '/Users/abbystevens/uchicago/dssg/flask-apps/basic_aequitas/uploaded_data'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
            filename = secure_filename(file.filename)
            uuid = 'customize'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], uuid))
            return redirect(url_for("uploaded_file", uuid=uuid))
    return render_template("file_upload.html")

    '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploaded_file/<uuid>', methods=['get', 'post'])
def uploaded_file(uuid):
    necessary_cols = ['model_id', 'entity_id', 'score', 'label_value', 'rank_abs', 'rank_pct']
    df = pd.read_csv(app.config['UPLOAD_FOLDER']+'/'+uuid)
    cols  = df.columns
    groups = []
    subgroups = {}
    for col in cols:
        if col not in necessary_cols:
            groups.append(col)
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

        configs = {}
        configs['reference_groups'] = subgroups

        if "ref_groups_method" in request.form:
            rgm = request.form["ref_groups_method"]
        else:
            rgm = "predefined"

        gv_df = audit(df, ref_groups_method=rgm, model_id=1, configs=configs, report=False)

        return gv_df.to_html()

    '''
    <!doctype html>
    <title>Good job</title>
    <h1>u did it</h1>
    '''

if __name__ == "__main__":
    app.run()
