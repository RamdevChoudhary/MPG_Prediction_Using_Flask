# This code is a Flask web application that uses a machine learning model to 
# predict the miles per gallon (mpg) of a car based on its features. Let's 
# go through the code and provide a detailed explanation of each part.


from flask import Flask, render_template, request
import joblib
from custom_transformers_mpg import FeatureAdder

import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd

# The code begins by importing the necessary libraries for the Flask web 
# application, loading a saved machine learning model, and importing 
# some data visualization libraries.
# --------------------------------------------------------------------------------

# -----------------------Transformers and Data Pipelines--------------------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline # sequential pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
# These lines import various classes and functions from scikit-learn for data 
# preprocessing and building data pipelines.
# --------------------------------------------------------------------------------

# ---------------------------Data Sampling----------------------------------------
from sklearn.model_selection import train_test_split
# This line imports the 'train_test_split' function from scikit-learn,
#  which will be used for splitting the data into training and testing sets.
# --------------------------------------------------------------------------------

# --------------------------- Machine Learning Models-----------------------------
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# These lines import different regression models from scikit-learn,
#  which will be used for training and predicting the car's mpg.
# --------------------------------------------------------------------------------

# -----------------------------Evaluation Metrices--------------------------------
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# These lines import evaluation metrics from scikit-learn,
#  which will be used to evaluate the performance of the trained models.


from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
# This line creates a Flask application instance.


pipeline = joblib.load("data_pipeline_mpg_v1.pkl")
model = joblib.load("random_forest_mpg_v1.model")
# These lines load a saved data pipeline and a trained machine learning
#  model from disk. The pipeline is used for data preprocessing, and 
# the model is used for predicting the car's mpg.

@app.route('/')
def home():
    return render_template('index.html')
# This is a route decorator that binds the root URL '/' to the home
#  function. When a user visits the root URL, the home function is 
# executed, and it returns the rendered template 'index.html'.

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the form
        cylinders = int(request.form['cylinders'])
        displacement = float(request.form['displacement'])
        horsepower = float(request.form['horsepower'])
        weight = float(request.form['weight'])
        acceleration = float(request.form['acceleration'])
        name = str(request.form['name'])
        model_year = int(request.form['model_year'])
        origin = str(request.form['origin'])

        # Prepare the input features as a DataFrame
        features = pd.DataFrame({
            'cylinders': [cylinders],
            'displacement': [displacement],
            'horsepower': [horsepower],
            'weight': [weight],
            'acceleration': [acceleration],
            'name': [name],
            'model_year': [model_year],
            'origin': [origin]
        })

        # Apply the same preprocessing steps used during training
        processed_features = pipeline.transform(features)

        # Make the prediction using the pre-trained model
        prediction = model.predict(processed_features)

        # Render the result template with the predicted mpg
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('error.html', error_message=str(e))
    
    # This is another route decorator that binds the URL /predict to the 
    # predict function. This function is executed when a POST request is 
    # made to the /predict URL. It retrieves the input features from the 
    # form submitted by the user, prepares them as a DataFrame, applies
    #  the same preprocessing steps used during training, makes a prediction
    #  using the pre-trained model, and renders the result.html template with
    #  the predicted mpg. If any error occurs during the process, the error.html 
    # template is rendered with the error message.

if __name__ == '__main__':
    app.run(debug=True)
    # Finally, this block of code ensures that the Flask application is run only 
    # if the script is executed directly (not imported as a module), and it runs 
    # the application in debug mode.


# Overall, this code sets up a Flask web application with two routes: '/' for the 
# home page and '/predict' for making predictions. The input features are collected
#  from a form submitted by the user, preprocessed using a data pipeline, and fed
#  into a machine learning model to predict the car's mpg. The predicted mpg is 
# then displayed to the user.
