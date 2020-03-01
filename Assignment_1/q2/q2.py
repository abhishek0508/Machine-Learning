import numpy as np
import random
import pandas as pd
import io
from collections import Counter
from sklearn.metrics import accuracy_score

train_url = 'https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q2/train.csv'
test_url = 'https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q2/test.csv'
test_label_url = "https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q2/test_labels.csv"

all_data = all_attributes = [
    {'b','c','x','f','k','s'},
    {'f','g','y','s'},
    {'n','b','c','g','r','p','u','e','w','y'},
    {'t','f'},
    {'a','l','c','y','f','m','n','p','s'},
    {'a','d','f','n'},
    {'c','w','d'},
    {'b','n'},
    {'k','n','b','h','g','r','o','p','u','e','w','y'},
    {'e','t'},
    {'b','c','u','e','z','r'},
    {'f','y','k','s'},
    {'f','y','k','s'},
    {'n','b','c','g','o','p','e','w','y'},
    {'n','b','c','g','o','p','e','w','y'},
    {'p','u'},
    {'n','o','w','y'},
    {'n','o','t'},
    {'c','e','f','l','n','p','s','z'},
    {'k','n','b','h','r','o','u','w','y'},  
    {'a','c','n','s','v','y'},
    {'g','l','m','p','u','w','d'}
    ]



class KNNClassifier:

  test_label = []
  train_label_data = []
  train_data_dummies = []
  
  def fun(self,dist,k):
      knn = []
      for i in range(0,k):
        knn.append(dist[i])
      
      # print(knn)
      knn_label = []
      for i in range(0,len(knn)):
        knn_label.append(knn[i][1])
      
      knn_value = Counter(knn_label)
      res = knn_value.most_common(1)[0][0]

      # print(res)
      return res

  def preprocessing(self,preprocess):
    # print("Before processing")

    preprocess = preprocess.replace('?',random.choice(['b','c','u','e','z','r']))
    # print("hererer i am  .................")
    preprocess = preprocess.to_numpy()
    dummy_insert_col = preprocess[0,:]

    length = 22
    for i in range(0,length):
      y = all_data[i] - set(preprocess[:,i])
      # print(y)
      list_y = list(y)
      for j in range(0,len(list_y)):
        dummy_insert_col[i]=list_y[j]
        preprocess = np.vstack([preprocess, dummy_insert_col])
    
    return preprocess

  def train(self,url):
    train = pd.read_csv(url)
    # print(train)
    self.train_label_data = train[train.columns[0]].to_numpy()
    # print(len(self.train_label_data))
    train.drop(train.columns[0],axis=1,inplace=True)
    # print(train.columns)
    train_after_processing = self.preprocessing(train)
    train_data_dummies = pd.get_dummies(pd.DataFrame(train_after_processing))
    self.train_data_dummies = train_data_dummies.to_numpy()
    # print(train_data_dummies)


  def predict(self,url):
    test = pd.read_csv(url)
    test_after_preprocessing = self.preprocessing(test)
    test_data_dummies = pd.get_dummies(pd.DataFrame(test_after_preprocessing))
    test_data_dummies = test_data_dummies.to_numpy()
    
    euclid_dist = []
    model_label = []
    model_label.append('e')
    print(self.train_label_data.shape[0])
    z=0
    for test_data_dummies_row in test_data_dummies:
      if(z<test.shape[0]):
        i=0
        for train_data_dummies_row in self.train_data_dummies:
          # print(train_data_dummies_row)
          if(i<self.train_label_data.shape[0]):
            row_diff = np.power((train_data_dummies_row - test_data_dummies_row),2)
            # print(np.sum(row_diff))
            row_euclid_diff = np.sqrt(np.sum(row_diff))
            # print(row_euclid_diff)
            x = self.train_label_data[i]
            # print(x)
            euclid_dist.append((row_euclid_diff,x))
            i=i+1
        
        # print(euclid_dist)
        
        dist = sorted(euclid_dist, key=lambda x: x[0])
        model_label.append(self.fun(dist,5))
        euclid_dist.clear()
        z=z+1

    return model_label


###########################code for running in google colab####################

# knn_classifier = KNNClassifier()
# knn_classifier = knc()
# knn_classifier.train(train_url)
# predictions = knn_classifier.predict(test_url)
# print(predictions)
# test_label = pd.read_csv(test_label_url)
# test_label = test_label.to_numpy();
# # print(test_label)
# test_label_new = []
# for i in range(0,test_label.shape[0]):
#   test_label_new.append(test_label[i][0])
# print(test_label_new)
# print (accuracy_score(test_label_new, predictions))