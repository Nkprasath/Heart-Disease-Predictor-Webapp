# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 08:52:38 2021

@author: naken
"""
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn import svm 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

st.header("Heart Diease Predictor")
st.subheader("This app predicts if a person has heart disease or not")

data = pd.read_csv("heart.csv")
st.write("Loaded heart.csv file......")    
st.subheader("""
Exploratory Data Analysis
""")
st.subheader('The dataframe')
st.write(data.head())

st.subheader('Describe dataframe')
st.write(data.describe())

st.write('Number of rows: {}, Number of columns: {}'.format(*data.shape))

#Separate Feature and Target Matrix
x = data.drop('target',axis = 1) 
y = data.target

# Split dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=109)

#Select model
st.subheader('Select you Model')
option = st.selectbox(
     'Models',
     ('SVM - Linear', 'SVM - rbf', 'SVM - poly', 'SVM - Sigmoid', 'Decision Tree' ,'Random Forest'))

st.write('You selected:', option)

st.header("Train Model")
if(option == 'SVM - Linear'):
    st.write("Training SVM - Linear model.....")
    ml = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    ml.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = ml.predict(x_test)
    score = ml.score(x_test,y_test)
    st.write("Accuracy of SVM - Linear is: ")
    st.success(score)
    st.write("Model Estimation")
    st.write(confusion_matrix(y_test,y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    st.pyplot()
    
if(option == 'SVM - rbf'):
    st.write("Training SVM - rbf model.....")
    ml = svm.SVC(kernel='rbf') # Linear Kernel
    #Train the model using the training sets
    ml.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = ml.predict(x_test)
    score = ml.score(x_test,y_test)
    st.write("Accuracy of SVM - rbf is: ")
    st.success(score)
    st.write("Model Estimation")
    st.write(confusion_matrix(y_test,y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    st.pyplot()
    
if(option == 'SVM - poly'):
    st.write("Training SVM - poly model.....")
    ml = svm.SVC(kernel='poly') # Linear Kernel
    #Train the model using the training sets
    ml.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = ml.predict(x_test)
    score = ml.score(x_test,y_test)
    st.write("Accuracy of SVM - poly is: ")
    st.success(score)
    st.write("Model Estimation")
    st.write(confusion_matrix(y_test,y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    st.pyplot()
    
if(option == 'SVM - Sigmoid'):
    st.write("Training SVM - Sigmoid model.....")
    ml = svm.SVC(kernel='sigmoid') # Linear Kernel
    #Train the model using the training sets
    ml.fit(x_train, y_train)
    #Predict the response for test dataset
    y_pred = ml.predict(x_test)
    score = ml.score(x_test,y_test)
    st.write("Accuracy of SVM - Sigmoid is: ")
    st.success(score)
    st.write("Model Estimation")
    st.write(confusion_matrix(y_test,y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    st.pyplot()
    

if(option == 'Decision Tree'):
    st.write("Training Decision Tree model.....")
    decTree_model = DecisionTreeClassifier()
    decTree_model.fit(x_train, y_train)
    #Predict the response for test dataset
    test_score2 = decTree_model.score(x_test, y_test)
    st.write("Accuracy of Decision Tree is: ")
    st.success(test_score2)
    st.write("Model Estimation")
    y_predictions = decTree_model.predict(x_test)
    conf_matrix = confusion_matrix(y_predictions, y_test)
    st.write(conf_matrix)
    fpr, tpr, thresholds = roc_curve(y_test, y_predictions)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--' )
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve') 
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    st.pyplot()
    
if(option == 'Random Forest'):
    st.write("Training Random Forest model.....")
    rf = RandomForestClassifier(n_estimators=100, max_depth=3)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    DT_score = rf.score(x_test, y_test)
    st.write("Accuracy of Random Forest is: ")
    st.success(DT_score)
    st.write("Model Estimation")
    conf_matrix = confusion_matrix(y_pred, y_test)
    st.write(conf_matrix)
    
    







