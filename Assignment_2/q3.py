import numpy as np
import pandas as pd
import io
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import preprocessing

url = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-3/airfoil.csv'
test_url = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_2/Datasets/Question-3/test.csv'
class Airfoil:

    optimal_params=[]
    scaler_object = []

    def preprocessing(self,df):
        scaler = MinMaxScaler()
        self.scaler_object = scaler.fit(df)
        df = self.scaler_object.transform(df)
        X = df[:,:5]
        y = df[:,5:6]
        X = np.hstack((np.ones((len(y),1)),X))
        # print(X.shape)
        # print(y.shape)
        return X,y
    
    def preprocessing_test(self,df):
        df = self.scaler_object.transform(df)
        X = df[:,:5]
        y = df[:,5:6]
        X = np.hstack((np.ones((len(y),1)),X))
        # print(X.shape)
        # print(y.shape)
        return X,y

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

    def predict(self,url):
        df = pd.read_csv(url)
        X_test,y_test = self.preprocessing_test(df)
        # print(X_test.shape)
        print(self.optimal_params.shape)
        predictions = []
        predictions =  X_test @ self.optimal_params
        return predictions

    def train(self,url):
        df = pd.read_csv(url)
        X,y = self.preprocessing(df)
        n_samples = len(y)
        n_features = np.size(X,1)
        params = np.zeros((n_features,1))
        n_iters = 3700
        learning_rate = 0.02 
        initial_cost = self.compute_cost(X, y, params)
        print("Initial cost is: ", initial_cost, "\n")
        (J_history, self.optimal_params) = self.gradient_descent(X, y, params, learning_rate, n_iters)
        print("Optimal parameters are: \n", self.optimal_params, "\n")
        print("Final cost is: ", J_history[-1])
        plt.plot(range(len(J_history)), J_history, 'r')
        plt.title("Convergence Graph of Cost Function")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Cost")
        plt.show()

# model3 = Airfoil()
# model3.train(url) # Path to the train.csv will be provided
# y_test,predictions = model3.predict(test_url)
# print(r2_score(y_test,predictions))
