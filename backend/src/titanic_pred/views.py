# our home page view
from django.shortcuts import render
import pickle
import os

from . import settings

def home(request):    
    return render(request, 'index.html')

# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    model_path = os.path.join(settings.BASE_DIR, "titanic_survival_ml_model.sav")
    scaler_path = os.path.join(settings.BASE_DIR, "scaler.sav")
    print("Model path:", model_path)
    print("Scaler path:", scaler_path)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return "Model or scaler file missing"

    with open(model_path, "rb") as model_file, open(scaler_path, "rb") as scaler_file:
        model = pickle.load(model_file)
        scaler = pickle.load(scaler_file)
    
    scaled_features = scaler.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]])
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)    
    
    if prediction[0] == 0:
        return "not survived", prediction_proba[0][0] * 100
    elif prediction[0] == 1:
        return "survived", prediction_proba[0][1] * 100
    else:
        return "error", None

# our result page view
def result(request):
    try:
        pclass = int(request.GET['pclass'])
        sex = int(request.GET['sex'])
        age = int(request.GET['age'])
        sibsp = int(request.GET['sibsp'])
        parch = int(request.GET['parch'])
        fare = float(request.GET['fare'])
        embC = int(request.GET['embC'])
        embQ = int(request.GET['embQ'])
        embS = int(request.GET['embS'])
    except (ValueError, KeyError) as e:
        return render(request, 'home.html', {'result': 'Invalid input data'})

    prediction_result = getPredictions(pclass, sex, age, sibsp, parch, fare, embC, embQ, embS)

    return render(request, 'home.html', {'result': prediction_result})
