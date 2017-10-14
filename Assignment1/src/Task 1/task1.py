import csv
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def linearReg(dataFrame):
    #remodle datetime column
    dataFrame['date'], dataFrame['time'] = dataFrame['pickup_datetime'].str.split(' ', 0).str #Split data and time into two features
    del dataFrame['pickup_datetime'] #remover old date+time feature
    del dataFrame['date'] #remove date
    dataFrame['time'] = pd.to_datetime(dataFrame['time']) #convert time to pandas datetime
    dataFrame['time'] = dataFrame['time'].dt.hour + dataFrame['time'].dt.minute/60 #converst datatime to float
    dataFrame['id'].replace(regex=True, inplace =True, to_replace=r'\D',value=r'')
    #print df
    #print '\n'


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



#NYCdataset = open("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", "r")
#NYCTripDuration = csv.reader(NYCdataset)
df100 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=100) #reading in CSV dataset
df500 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=500)
df1000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=1000)
df5000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=5000)
df10000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=10000)
df50000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=50000)
df100000 = pd.read_csv("/Users/HughLavery/Documents/College/Year 4/CS4404/New York City Taxi Trip Duration/train.csv", nrows=100000)

linearReg(df100)
linearReg(df500)
linearReg(df1000)
linearReg(df5000)
linearReg(df10000)
linearReg(df50000)
linearReg(df100000)



