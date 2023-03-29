from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict_cancer():
    Universal_mean=eval(request.form.get("Universal_mean"))
    universal_se=eval(request.form.get("universal_se"))
    universal_worst=eval(request.form.get("universal_worst"))
    
    result=model.predict(np.array([[Universal_mean,universal_se,universal_worst]]))
    
    if result[0]==1:
        return "<h1 style='color:green'>person is Suffering by 'M' type of breast cancer</h1>"
    else:
        return "<h1 style='color:red'>person is Suffering by 'B' type of breast cancer</h1>"
    
    
# @app.route("/predict",methods=["GET"])
# def predict_placement():
#     cgpa=float(request.args.get("cgpa"))
#     iq=float(request.args.get("iq"))
#     profile_score=float(request.args.get("profile_score"))
    
    
#     result=model.predict(np.array([[cgpa,iq,profile_score]]))
    
#     if result[0]==1:
#         return "<h1 style='color:green'>PLACED</h1>"
#     else:
#         return "<h1 style='color:red'>NOT PLACED</h1>"   

app.run(host="0.0.0.0",port=8080)