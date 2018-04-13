from flask import (
    Flask,
    Markup,
    request,
    redirect,
    url_for,
    flash,
    render_template,
    session,
)
from flask_bootstrap import Bootstrap
import pandas as pd

from aequitas.fairness import Fairness
from aequitas.preprocessing import preprocess_input_df

from aequitas_cli.aequitas_audit import audit
from aequitas_cli.utils.configs_loader import Configs

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'])
sample_data = {'sample1': 'app/sample_data/sample1.csv',
               'sample2': 'app/sample_data/sample2.csv',
               'sample3': 'app/sample_data/sample1.csv'}

app = Flask(__name__)
app.secret_key = 'super secret key'
Bootstrap(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        session.clear()
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
            df.to_csv('tmp.csv', index=False)
            filetype = 'upload'
            # session['df'] = df.to_json()
            return redirect(url_for("uploaded_file", filetype=filetype))
    return render_template("file_upload.html")


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template("about.html")


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    return render_template("contact.html")


@app.route('/customize/<filetype>', methods=['get', 'post'])
def uploaded_file(filetype):
    if filetype == 'custom':
        df = pd.read_csv('tmp.csv')
    else:
        df = pd.read_csv(sample_data[filetype])
    df, groups = preprocess_input_df(df)
    subgroups = {}
    for col in groups:
        subgroups[col] = (list(set(df[col])))

    if "submit" not in request.form:
        f = Fairness()
        fairness_measures = f.fair_measures_supported
        return render_template("customize_report2.html",
                                           categories=groups,
                                           subcategories=subgroups,
                                           fairness=fairness_measures)
    else:
        print(request.form)
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
        fairness_measures = request.form.getlist('fairness_measures')
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

        gv_df, report = audit(df, model_id=1, configs=configs, preprocessed=True)
        content = Markup(report)
        # os.remove('tmp.csv')
        return render_template('report.html', content=content)

if __name__ == "__main__":
    app.run()
