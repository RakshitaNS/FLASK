from flask import Flask , render_template,request
import numpy as np 
import pandas as pd 
import os 
import pickle
from src.pipeline.predict_pipeline import predictpipeline

application = Flask(__name__)#start of my application 




@application.route('/')
def home():
    return render_template("home.html")

@application.route('/predict_ap',methods=['POST'])

#capture what is present inside 'data' (the input we are going to give ) using request in json format 
#and store in data variable 
#so as soon as u hit the predict.api ur input will be loaded using request in json format 

    #print('ticket price is {a}'.format(output))
def predict_ap():
    data=[x for x in request.form.values()]
    t = pd.DataFrame(data)
    pr =predictpipeline()

    output = pr.predic(t)
    return render_template("Home.html",prediction_text="{}".format(output))
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    application.run(debug=True)
