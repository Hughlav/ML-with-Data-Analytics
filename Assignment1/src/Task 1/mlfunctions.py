import csv
import sklearn
import pandas as pd
import math
import mlfunctions as mlf
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import svm
from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.model_selection import cross_val_score

def remodelDataNYC(dataFrame):
    #remodle datetime column
    dataFrame['date'], dataFrame['time'] = dataFrame['pickup_datetime'].str.split(' ', 0).str #Split data and time into two features
    del dataFrame['pickup_datetime'] #remover old date+time feature
    del dataFrame['date'] #remove date
    del dataFrame['id']
    dataFrame['time'] = pd.to_datetime(dataFrame['time']) #convert time to pandas datetime
    dataFrame['time'] = dataFrame['time'].dt.hour + dataFrame['time'].dt.minute/60 #converst datatime to float

    return dataFrame

def remodelDataSUM(dataFrame):
    #remodle datetime column
    del dataFrame['Instance'] #remove instance
    del dataFrame['Target Class']
    return dataFrame

def remodelDataSUMN(dataFrame):
    #remodle datetime column
    del dataFrame['Instance'] #remove instance
    del dataFrame['Noisy Target Class']
    return dataFrame

def linearReg(dataFrame, yVar):
    # Split data into X and and Y
    X = dataFrame.drop(yVar, axis=1)
    y = dataFrame [[yVar]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    #Train Model
    regression_model = LinearRegression()
    regression_model.fit(X_train.values, y_train.values)
    
    #Evaluate
    evaluateModelReg(regression_model,X,y,y_test,X_test)
    


def randomForestRegression(dataFrame, yVar):
    
    # Split data into X and and Y
    X = dataFrame.drop(yVar, axis=1)
    y = dataFrame [[yVar]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    #Standardising data
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)
    
    #Fitting random forsest regressor
    rfr = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
    rfr.fit(X_train, y_train.values.ravel())

    #evaluate
    evaluateModelReg(rfr,X,y,y_test,X_test)

   
def logisticReg(dataFrame, yVar):
    # Split data into X and and Y
    X = dataFrame.drop(yVar, axis=1)
    y = dataFrame [[yVar]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    #Train Model
    lr = LogisticRegression()
    lr.fit(X_train,y_train.values.ravel())
    
    #evaluate
    evaluateModelLog(lr,X,y,y_test,X_test)

def linearSVC(dataFrame, yVar):
    # Split data into X and and Y
    X = dataFrame.drop(yVar, axis=1)
    y = dataFrame [[yVar]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    #Train Model
    svc = svm.SVC(kernel='linear', C=1.0)
    svc.fit(X_train,y_train.values.ravel())

    #evaluate
    evaluateModelLog(svc,X,y,y_test,X_test)

def createDummiesNYC(dataFrame):
    #cut trip duration and time
    dataFrame['time'] = pd.cut(dataFrame['time'], [0,5,10,15,20,24], labels=[1,2,3,4,5])
    dataFrame['trip_duration'] = pd.cut(dataFrame['trip_duration'], [0,2500,5000,7500,10000,3600000], labels=[1,2,3,4,5])
    dataFrame['passenger_count'] = pd.cut(dataFrame['passenger_count'], [0,1,2,3,4,6], labels=[1,2,3,4,5])
    dummyvid = pd.get_dummies(dataFrame['vendor_id'], prefix='vendor_id')
    dataFrame = dataFrame[['trip_duration', 'time', 'passenger_count']].join(dummyvid.ix[:, 'vendor_id_2':])


    #Getting rid of NaNs
    #dataFrame = dataFrame.cat.add_categories([1])
    dataFrame = dataFrame.fillna(1)
    return dataFrame
    
def createDummiesSUM(dataFrame, maximum):
    #cut features into categories
    cols = [col for col in dataFrame.columns]
    
    for cols in dataFrame:
        dataFrame[cols] = pd.cut(dataFrame[cols], [0,50,5000,50000,500000,maximum], labels=[1,2,3,4,5])
    
    #Getting rid of NaNs
    #dataFrame = dataFrame.cat.add_categories([1])
    #dataFrame = dataFrame.fillna(1)
    return dataFrame


def evaluateModelReg(regression_model,X,y,y_test,X_test):
    #10 fold cross validation
    scores = cross_val_score(regression_model,X,y.values.ravel(),cv=10)
    print "10 fold cross vaidation scores"
    print np.mean(scores)

    #Calculating RMSE for Evaluation
    expect = y_test.values.ravel()
    predict = regression_model.predict(X_test)
    MSE = mean_squared_error(expect,predict)
    RMSE = sqrt(MSE)
    print "printing RMSE"
    print RMSE

    #Calculating MEA (mean absolute error)
    MEA = mean_absolute_error(expect,predict)
    print "Mean absolute error: "
    print MEA
    print "\n\n"

def evaluateModelLog(classification_model,X,y,y_test,X_test):
    #10 fold cross validation
    scores = cross_val_score(classification_model,X,y.values.ravel(),cv=10)
    print "10 fold cross vaidation scores"
    print np.mean(scores)

    #Accuracy
    expect = y_test.values.ravel()
    predict = classification_model.predict(X_test)
    accuracy = accuracy_score(expect, predict)
    print "Accuracy is: "
    print accuracy

    #Precision weighted
    precision = precision_score(expect, predict, average='weighted')
    print "Precision is: "
    print precision
    print "\n"

