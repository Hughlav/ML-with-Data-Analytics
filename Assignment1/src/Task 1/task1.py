import csv
import sklearn
import pandas as pd
import math
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

def remodelData(dataFrame):
    #remodle datetime column
    dataFrame['date'], dataFrame['time'] = dataFrame['pickup_datetime'].str.split(' ', 0).str #Split data and time into two features
    del dataFrame['pickup_datetime'] #remover old date+time feature
    del dataFrame['date'] #remove date
    del dataFrame['id']
    dataFrame['time'] = pd.to_datetime(dataFrame['time']) #convert time to pandas datetime
    dataFrame['time'] = dataFrame['time'].dt.hour + dataFrame['time'].dt.minute/60 #converst datatime to float
    #print dataFrame
    #print '\n'
    return dataFrame

def linearReg(dataFrame):
    # Split data into X and and Y
    X = dataFrame.drop('trip_duration', axis=1)
    y = dataFrame [['trip_duration']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    #Train Model
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    
    #Evaluate
    evaluateModelReg(regression_model,X,y,y_test,X_test)
    


def randomForestRegression(dataFrame):
    
    # Split data into X and and Y
    X = dataFrame.drop('trip_duration', axis=1)
    y = dataFrame [['trip_duration']]
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

   
def logisticReg(dataFrame):
    # Split data into X and and Y
    X = dataFrame.drop('trip_duration', axis=1)
    y = dataFrame [['trip_duration']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    #Train Model
    lr = LogisticRegression()
    lr.fit(X_train,y_train.values.ravel())
    
    #evaluate
    #evaluateModelLog(lr,X,y,y_test,X_test)

def linearSVC(dataFrame):
    # Split data into X and and Y
    X = dataFrame.drop('trip_duration', axis=1)
    y = dataFrame [['trip_duration']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    #Train Model
    svc = svm.SVC(kernel='linear', C=1.0)
    svc.fit(X_train,y_train)

    #evaluate
    #evaluateModelLog(svc,X,y,y_test,X_test)

def createDummiesNYC(dataFrame):
    #cut trip duration and time
    dataFrame['time'] = pd.cut(dataFrame['time'], [0,5,10,15,20,24], labels=[1,2,3,4,5])
    dataFrame['trip_duration'] = pd.cut(dataFrame['trip_duration'], [0,2500,5000,7500,10000,3600000], labels=[1,2,3,4,5])
    dataFrame['passenger_count'] = pd.cut(dataFrame['passenger_count'], [0,1,2,3,4,6], labels=[1,2,3,4,5])
    dummyvid = pd.get_dummies(dataFrame['vendor_id'], prefix='vendor_id')
    dataFrame = dataFrame[['trip_duration', 'time', 'passenger_count']].join(dummyvid.ix[:, 'vendor_id_2':])
    print dataFrame.head()
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


#NYCdataset = open("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", "r")
#NYCTripDuration = csv.reader(NYCdataset)
df100 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=100) #reading in CSV dataset
#df500 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=500)
#df1000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=1000)
#df5000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=5000)
#df10000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=10000)
#df50000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=50000)
df100000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=100000)
#df500k = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=500000)
#df1m = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=1000000)


df100 = remodelData(df100)
#df500 = remodelData(df500)
#df1000 = remodelData(df1000)
#df5000 = remodelData(df5000)
#df10000 = remodelData(df10000)
#df50000 = remodelData(df50000)
df100000 = remodelData(df100000)
#df500k = remodelData(df500k)
#df1m = remodelData(df1m)


print "data remodled"

#working!
#linearReg(df100)
#linearReg(df500)
#linearReg(df1000)
#linearReg(df5000)
#linearReg(df10000)
#linearReg(df50000)
linearReg(df100000)
#linearReg(df500k)
#linearReg(df1m)

print "linear regression done"

#randomForestRegression(df100)
#randomForestRegression(df500)
#randomForestRegression(df1000)
#randomForestRegression(df5000)
#randomForestRegression(df10000)
#randomForestRegression(df50000)
randomForestRegression(df100000)
#randomForestRegression(df500k)
#randomForestRegression(df1m)

print "random forest done"

npany = np.any(np.isnan(df100000))
print npany
npall = np.all(np.isfinite(df100000))
print npall

df100 = createDummiesNYC(df100)
#df500 = createDummiesNYC(df500)
#df1000 = createDummiesNYC(df1000)
#df5000 = createDummiesNYC(df5000)
#df10000 = createDummiesNYC(df10000)
#df50000 = createDummiesNYC(df50000)
df100000 = createDummiesNYC(df100000)
#df500k = createDummiesNYC(df500k)
#df1m = createDummiesNYC(df1m)

logisticReg(df100)
#logisticReg(df500)
#logisticReg(df1000)
#logisticReg(df5000)
#logisticReg(df10000)
#logisticReg(df50000)
logisticReg(df100000)
#logisticReg(df500k)
#logisticReg(df1m)



linearSVC(df100)
#linearSVC(df500)
#linearSVC(df1000)
#linearSVC(df5000)
#linearSVC(df10000)
#linearSVC(df50000)
linearSVC(df100000)
#linearSVC(df500k)
#linearSVC(df1m)

print "done"