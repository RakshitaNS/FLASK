
import os 
import numpy as np 
import pandas as pd 
import os 
import pickle 


class predictpipeline:
    def __init__(self):
        pass
    def predic(self,inp):
        model = pickle.load(open('model.pkl','rb'))
        preprocess = pickle.load(open('preprocessor.pkl','rb'))
        transformed_data = preprocess.fit_transform(inp)
    #print(np.array(list(data.values())).reshape(1,-1)) #1,-1 to say ur values are single input 
        output = model.predict(np.array(transformed_data).reshape(1,-1))
        return output 
    
