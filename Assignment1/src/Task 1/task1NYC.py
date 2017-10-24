import csv
import mlfunctions as mlf
import pandas as pd
import numpy as np

csv_file = "/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv"
yVar = 'trip_duration'

df100 = pd.read_csv(csv_file, nrows=100) #reading in CSV dataset
#df500 = pd.read_csv(csv_file, nrows=500)
#df1000 = pd.read_csv(csv_file, nrows=1000)
#df5000 = pd.read_csv(csv_file, nrows=5000)
#df10000 = pd.read_csv(csv_file, nrows=10000)
#df50000 = pd.read_csv(csv_file, nrows=50000)
df100000 = pd.read_csv(csv_file, nrows=100000)
#df500k = pd.read_csv(csv_file, nrows=500000)
#df1m = pd.read_csv(csv_file, nrows=1000000)

print df100.head()

df100 = mlf.remodelDataNYC(df100)
#df500 = remodelDataNYC(df500)
#df1000 = remodelDataNYC(df1000)
#df5000 = remodelDataNYC(df5000)
#df10000 = remodelDataNYC(df10000)
#df50000 = remodelDataNYC(df50000)
#df100000 = mlf.remodelDataNYC(df100000)
#df500k = remodelDataNYC(df500k)
#df1m = remodelDataNYC(df1m)


print "Linear Regression"

#working!
mlf.linearReg(df100, yVar)
#mlf.linearReg(df500)
#mlf.linearReg(df1000)
#mlf.linearReg(df5000)
#mlf.linearReg(df10000)
#mlf.linearReg(df50000)
#mlf.linearReg(df100000)
#mlf.linearReg(df500k)
#mlf.linearReg(df1m)

print "Random forest regression"

mlf.randomForestRegression(df100,yVar)
#mlf.randomForestRegression(df500)
#mlf.randomForestRegression(df1000)
#mlf.randomForestRegression(df5000)
#mlf.randomForestRegression(df10000)
#mlf.randomForestRegression(df50000)
#mlf.randomForestRegression(df100000)
#mlf.randomForestRegression(df500k)
#mlf.randomForestRegression(df1m)

print df100.head()

df100 = mlf.createDummiesNYC(df100)
#df500 = createDummiesNYC(df500)
#mlf.df1000 = createDummiesNYC(df1000)
#mlf.df5000 = createDummiesNYC(df5000)
#mlf.df10000 = createDummiesNYC(df10000)
#mlf.df50000 = createDummiesNYC(df50000)
#df100000 = mlf.createDummiesNYC(df100000)
#mlf.df500k = createDummiesNYC(df500k)
#mlf.df1m = createDummiesNYC(df1m)

print "Logistic Regression"

mlf.logisticReg(df100,yVar)
#mlf.logisticReg(df500)
#mlf.logisticReg(df1000)
#mlf.logisticReg(df5000)
#mlf.logisticReg(df10000)
#mlf.logisticReg(df50000)
#mlf.logisticReg(df100000,yVar)
#mlf.logisticReg(df500k)
#mlf.logisticReg(df1m)

print "Linear SVC"

mlf.linearSVC(df100,yVar)
#mlf.linearSVC(df500)
#mlf.linearSVC(df1000)
#mlf.linearSVC(df5000)
#mlf.linearSVC(df10000)
#mlf.linearSVC(df50000)
#mlf.linearSVC(df100000,yVar)
#mlf.linearSVC(df500k)
#mlf.linearSVC(df1m)

print df100.head()

print "done"