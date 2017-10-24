import csv
import mlfunctions as mlf
import pandas as pd
import numpy as np

csv_path = "/Users/HughLavery/Documents/College/Year 4/CS4404/YearPredictionMSD.csv"
yVar = 'Year'

df100 = pd.read_csv(csv_path, nrows=100)

df100 = mlf.scaling(df100)


print "Linear Regression"
mlf.linearReg(df100, yVar)

print "RFR"
mlf.randomForestRegression(df100,yVar)

print "Creating Classes"
df100 = mlf.createDummiesMSD(df100)
print df100.head()

print "Logistic Regression"
mlf.logisticReg(df100,yVar)

print "linear SVC"
mlf.linearSVC(df100,yVar)