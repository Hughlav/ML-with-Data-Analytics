import csv
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

def remodelData(dataFrame):
    #remodle datetime column
    dataFrame['date'], dataFrame['time'] = dataFrame['pickup_datetime'].str.split(' ', 0).str #Split data and time into two features
    del dataFrame['pickup_datetime'] #remover old date+time feature
    del dataFrame['date'] #remove date
    dataFrame['time'] = pd.to_datetime(dataFrame['time']) #convert time to pandas datetime
    dataFrame['time'] = dataFrame['time'].dt.hour + dataFrame['time'].dt.minute/60 #converst datatime to float
    dataFrame['id'].replace(regex=True, inplace =True, to_replace=r'\D',value=r'')
    #print df
    #print '\n'
    return dataFrame

def linearReg(dataFrame):
    # Split data into X and and Y
    X = dataFrame.drop('trip_duration', axis=1)
    y = dataFrame [['trip_duration']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    #Train Model

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    #Coefficients

    for idx, col_name in enumerate(X_train.columns):
        print ("The Coefficient for {} is {}".format(col_name,regression_model.coef_[0][idx]))
    

    intercept = regression_model.intercept_[0]
    print("The intercept for our model is {}".format(intercept))
    a = regression_model.score(X_test, y_test)
    print a
    print '\n'

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
    rfr = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
    rfr.fit(X_train, y_train.values.ravel())
    print y_train

    #R squared score
    #predicted_train = rfr.predict(X_train)
    #predicted_test = rfr.predict(X_test)
    #test_score = r2_score(y_test, predicted_test)
    
    
    #print 'Test data R-2 score:{}', test_score
   
def logisticReg(dataFrame):
    # Split data into X and and Y
    X = dataFrame.drop('trip_duration', axis=1)
    y = dataFrame [['trip_duration']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    
    #Train Model
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    
    #predictions
    expect = y_test
    predict = lr.predict(X_test)

    #summarise fit
    print (metrics.classification_report(expect,predict))
    print(metrics.confusion_matrix(expect,predict))



#NYCdataset = open("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", "r")
#NYCTripDuration = csv.reader(NYCdataset)
df100 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=100) #reading in CSV dataset
df500 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=500)
df1000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=1000)
df5000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=5000)
df10000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=10000)
df50000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=50000)
df100000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=100000)

df100 = remodelData(df100)
df500 = remodelData(df500)
df1000 = remodelData(df1000)
df5000 = remodelData(df5000)
df10000 = remodelData(df10000)
df50000 = remodelData(df50000)
df100000 = remodelData(df100000)

linearReg(df100)
linearReg(df500)
linearReg(df1000)
linearReg(df5000)
linearReg(df10000)
linearReg(df50000)
linearReg(df100000)

#randomForestRegression(df100)
#randomForestRegression(df500)
#randomForestRegression(df1000)
#randomForestRegression(df5000)
#print "5k"
#randomForestRegression(df10000)
#print "10k"
#randomForestRegression(df50000)
#print "50k"
#randomForestRegression(df100000)
#print "100k"

logisticReg(df100)
logisticReg(df500)
logisticReg(df1000)
logisticReg(df5000)
logisticReg(df10000)
logisticReg(df50000)
logisticReg(df100000)

print "done"