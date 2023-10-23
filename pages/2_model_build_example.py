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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns


def model_building():
    col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset
    pre_proc_dataset = pd.read_csv("datasets/diabetes.csv", header=1, names=col_names)
    pre_proc_dataset
    #split dataset in features and target variable
    feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
    X = pre_proc_dataset[feature_cols] # Features
    y = pre_proc_dataset.label # Target variable
   # split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

    # Fitting Logistic Regression to the Training set
    logreg = LogisticRegression(random_state=16)

    logreg.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = logreg.predict(X_test)
    # Predict probabilities
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Plot confusion matrix

    # confusion matrix sns heatmap 
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.Text(0.5,257.44,'Predicted label');
    fig
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.show()

st.set_page_config(page_title="Model Building Example", page_icon="🌍")
st.markdown("# Model Building Example")
st.sidebar.header("Model Building Example")
st.write(
    """This is an example where we can create a logistic regression model against some data."""
)

model_building()

show_code(model_building)
