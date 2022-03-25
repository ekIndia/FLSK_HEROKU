from model import InputForm
from flask import Flask, render_template, request
from compute import compute
import sys

try:
    template_name = sys.argv[1]
except IndexError:
    template_name = 'view1'

app = Flask(__name__)

#@app.route('/vib1', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    rr="Nothing"
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        rr="CH"
        #if form.by.data.split(==0:
            #rr="CH"
        result = compute(form.R.data, form.Q.data,
                         form.bx.data,form.by.data,form.M.data)
    else:
        result = None
    return render_template('view1.html',form=form, result=result, rr=rr)
    #return render_template(template_name + '.html',

if __name__ == '__main__':
    app.run()
