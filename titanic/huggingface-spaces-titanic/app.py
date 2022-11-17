import gradio as gr
import numpy as np
from PIL import Image
import requests
import xgboost

import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(sex, age, pclass, embarked, parch):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(parch)
    input_list.append(embarked)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.       
    return res[0]
        
demo = gr.Interface(
    fn=titanic,
    title="Titanic Survival Predictive Analytics",
    description="Experiment with different entries to predict if the person will survive.",
    allow_flagging="never",
    ### Create user interface with 4 text boxes
    inputs=[
        gr.inputs.Number(default=1.0, label="sex (1 for female and 0 for male)"),
        gr.inputs.Number(default=1.0, label="age (in years)"),
        gr.inputs.Number(default=1.0, label="pclass "),
        gr.inputs.Number(default=1.0, label="parch "),
        gr.inputs.Number(default=1.0, label="embarked (2 for c, 1 for s and 0 for q)"),
        ],
    outputs=gr.inputs.Number(label="survived"))

demo.launch()

