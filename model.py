from wtforms import Form, FloatField, validators, IntegerField, StringField
from math import pi

class InputForm(Form):
    R = StringField(
        label='R1,R2,R3', default="3.446,2.488,2.4714",
        validators=[validators.InputRequired()])
    Q = StringField(
        label='Q1,Q2,Phi', default="0.8619,-0.2519,3.14285",
        validators=[validators.InputRequired()])
    bx = StringField(
        label='x_mn,x_mx,N,x_indx', default="3,4,10,0",
        validators=[validators.InputRequired()])
    by = StringField(
        label='y_mn,y_mx,N,y_indx', default="0.0,3.2,10,5",
        validators=[validators.InputRequired()])
    M = StringField(
        label='Molecule', default="HFCO",
        validators=[validators.InputRequired()])
