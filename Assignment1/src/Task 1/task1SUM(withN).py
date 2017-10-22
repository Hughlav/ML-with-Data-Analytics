import csv
import mlfunctions as mlf
import pandas as pd
import numpy as np

csv_path = "/Users/HughLavery/Documents/College/Year 4/CS4404/The SUM dataset/The SUM dataset, with noise.csv"
yVar = 'Noisy Target'

df100 = pd.read_csv(csv_path, nrows=100, sep=';')
df500 = pd.read_csv(csv_path, nrows=500, sep=';')
df1000 = pd.read_csv(csv_path, nrows=1000, sep=';')
df5000 = pd.read_csv(csv_path, nrows=5000, sep=';')
df10000 = pd.read_csv(csv_path, nrows=10000, sep=';')
df50000 = pd.read_csv(csv_path, nrows=50000, sep=';')
df100k = pd.read_csv(csv_path, nrows=100000, sep=';')
df500k = pd.read_csv(csv_path, nrows=500000, sep=';')
df1m = pd.read_csv(csv_path, nrows=968135, sep=';')
print df1m.head()

maximum = df1m['Noisy Target'].max()
dfs = [df100,df500,df1000 ,df5000,df10000,df50000,df100k,df500k,df1m]

for dfs in dfs:
    dfs = mlf.remodelDataSUMN(dfs)


print "Linear Regression"
mlf.linearReg(df100, yVar)
mlf.linearReg(df500, yVar)
mlf.linearReg(df1000, yVar)
mlf.linearReg(df5000, yVar)
mlf.linearReg(df10000, yVar)
mlf.linearReg(df50000, yVar)
mlf.linearReg(df100k, yVar)
mlf.linearReg(df500k, yVar)
mlf.linearReg(df1m, yVar)

print "RFR"
mlf.randomForestRegression(df100,yVar)
mlf.randomForestRegression(df500, yVar)
mlf.randomForestRegression(df1000, yVar)
mlf.randomForestRegression(df5000, yVar)
mlf.randomForestRegression(df10000, yVar)
mlf.randomForestRegression(df50000, yVar)
mlf.randomForestRegression(df100k, yVar)
mlf.randomForestRegression(df500k, yVar)
mlf.randomForestRegression(df1m, yVar)

print "Creating Classes"
df100 = mlf.createDummiesSUM(df100,maximum)
df500 = mlf.createDummiesSUM(df500,maximum)
df1000 = mlf.createDummiesSUM(df1000,maximum)
df5000 = mlf.createDummiesSUM(df5000,maximum)
df10000 = mlf.createDummiesSUM(df10000,maximum)
df50000 = mlf.createDummiesSUM(df50000,maximum)
df100k = mlf.createDummiesSUM(df100k,maximum)
df500k = mlf.createDummiesSUM(df500k,maximum)
df1m = mlf.createDummiesSUM(df1m,maximum)


print "Logistic Regression"
mlf.logisticReg(df100,yVar)
mlf.logisticReg(df500, yVar)
mlf.logisticReg(df1000, yVar)
mlf.logisticReg(df5000, yVar)
mlf.logisticReg(df10000, yVar)
mlf.logisticReg(df50000, yVar)
mlf.logisticReg(df100k, yVar)
mlf.logisticReg(df500k, yVar)
mlf.logisticReg(df1m, yVar)

print "linear SVC"
mlf.linearSVC(df100,yVar)
mlf.linearSVC(df500, yVar)
mlf.linearSVC(df1000, yVar)
mlf.linearSVC(df5000, yVar)
mlf.linearSVC(df10000, yVar)
mlf.linearSVC(df50000, yVar)
mlf.linearSVC(df100k, yVar)
mlf.linearSVC(df500k, yVar)
mlf.linearSVC(df1m, yVar)

