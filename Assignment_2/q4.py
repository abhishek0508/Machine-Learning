import numpy as np
import pandas as pd
import io
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# url='https://raw.githubusercontent.com/abhishek0508/Machine-Learning-Assignments/master/Assignment_2/Datasets/Question-5/Train(1).csv'

class Weather:

    optimal_params = None

    def preprocessing(self,df):
        # dropped formatted date as there is very litter correlation between predicted value and date  
        df['Hour'] = df['Formatted Date'].apply(pd.to_datetime, utc=True).dt.hour
        df['Year'] = df['Formatted Date'].apply(pd.to_datetime, utc=True).dt.year
        df['month'] = df['Formatted Date'].apply(pd.to_datetime, utc=True).dt.month
        df['date'] = df['Formatted Date'].apply(pd.to_datetime, utc=True).dt.day
        

        df = df.drop(labels=['Formatted Date'],axis=1)
        # df = df.drop(labels=['Formatted Date'],axis=1)
        # similar reason as above
        df = df.drop(labels=['Daily Summary'],axis=1)
#         print(df.shape)
        df = df.drop(labels=['Summary'],axis=1)
#         print(df.shape)
        # df["Precip Type"].fillna("No_Type", inplace = True) 
#         print(df)
        df = df.drop(labels=['Precip Type'],axis=1) 
        # print(df.shape)
        return df

    def compute_cost(self, X, y, params):
        n_samples = len(y)
        h = X @ params
        return (1/(2*n_samples))*np.sum((h-y)**2)
    
    def gradient_descent(self, X, y, params, learning_rate, n_iters):
        n_samples = len(y)
        J_history = np.zeros((n_iters,1))

        for i in range(n_iters):
            params = params - (learning_rate/n_samples) * X.T @ (X @ params - y) 
            J_history[i] = self.compute_cost(X, y, params)

        return (J_history, params)

    def train(self,url):
        df = pd.read_csv(url)
        df = self.preprocessing(df)
        
        y_train = df['Apparent Temperature (C)'].to_numpy().reshape(-1,1)
        X_train = df.drop(labels=['Apparent Temperature (C)'],axis=1).to_numpy()

        n_samples = len(y_train)
        
        X_train = np.hstack((np.ones((n_samples,1)),X_train))
        
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        
        n_features = np.size(X_train,1)
        params = np.zeros((n_features,1))

        n_iters = 3300
        learning_rate = 0.002  
        initial_cost = self.compute_cost(X_train, y_train, params)
        print("Initial cost is: ", initial_cost, "\n")
        (J_history, self.optimal_params) = self.gradient_descent(X_train, y_train, params, learning_rate, n_iters)
        print("Optimal parameters are: \n", self.optimal_params, "\n")
        print("Final cost is: ", J_history[-1])
        plt.plot(range(len(J_history)), J_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()


    def predict(self,url):
        df = pd.read_csv(url)
        # y_test = df['Apparent Temperature (C)'].to_numpy()
        # df = df.drop(labels=['Apparent Temperature (C)'],axis=1)
        X_test = self.preprocessing(df).to_numpy()
        X_test = np.hstack((np.ones((len(y_test),1)),X_test))
        scaler = MinMaxScaler()
        scaler.fit(X_test)
        X_test = scaler.transform(X_test)
        # print("X_shape=",X_test.shape)
        # print(len(self.optimal_params))
        predictions = []
        predictions =  X_test @ self.optimal_params
        predictions.reshape(-1,1).T
        return predictions
    
    def mean_absolute_percentage_error(self,y_true,y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        

# wr = Weather()
# wr.train('/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-4/weather.csv') # Path to the train.csv will be provided
# y_test,predictions = wr.predict('/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-4/test.csv')
# print("R2-Score == ",r2_score(y_test,predictions))
# print("Mean Squared Error",mean_squared_error(y_test,predictions))
# print("Mean Absolute Error",mean_absolute_error(y_test,predictions))
# y_test = y_test.reshape(-1,1).T
# predictions = predictions.reshape(-1,1).T
# print("Mean Absolute Percentage Error",wr.mean_absolute_percentage_error(y_test,predictions))
