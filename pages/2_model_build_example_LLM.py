# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from urllib.error import URLError

import pydeck as pdk

import streamlit as st
from streamlit.hello.utils import show_code

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns


def model_building():
    col_names = ["Id","SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
    # load dataset
    pre_proc_dataset = pd.read_csv("datasets/Iris.csv", header=1, names=col_names)
    pre_proc_dataset.head(5)

    st.write(pre_proc_dataset.count())
    
    pre_proc_dataset['Species'].value_counts()

    #split dataset in features and target variable
    X=pre_proc_dataset.iloc[:,:5]
    y=pre_proc_dataset.iloc[:,5]  

   # split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Fitting Logistic Regression to the Training set
    logreg = LogisticRegression()

    logreg.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = logreg.predict(X_test)
    # Predict probabilities

    confusion_matrix(y_test,y_pred)

    accuracy=accuracy_score(y_test,y_pred)*100
    st.write("Accuracy of the model is {:.2f}".format(accuracy))


st.set_page_config(page_title="Model Building Example", page_icon="🌍")
st.markdown("# Model Building Example")
st.sidebar.header("Model Building Example")
st.write(
    """This is an example where we can create a logistic regression model against some data !!!"""
)

model_building()

show_code(model_building)
