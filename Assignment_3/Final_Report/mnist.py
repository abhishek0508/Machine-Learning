import numpy as np
import idx2numpy
import sys
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn import preprocessing
from collections import Counter

len_dir = len(sys.argv[1])
directory = sys.argv[1]

x_test = None
y_test = None
x_train = None
y_train = None

if(directory[len_dir-1]!='/'):
    x_test = idx2numpy.convert_from_file(directory+"/t10k-images-idx3-ubyte")
    y_test = idx2numpy.convert_from_file(directory+"/t10k-labels-idx1-ubyte")
    x_train = idx2numpy.convert_from_file(directory+"/train-images-idx3-ubyte")
    y_train = idx2numpy.convert_from_file(directory+"/train-labels-idx1-ubyte")
else:
    x_test = idx2numpy.convert_from_file(directory+"t10k-images-idx3-ubyte")
    y_test = idx2numpy.convert_from_file(directory+"t10k-labels-idx1-ubyte")
    x_train = idx2numpy.convert_from_file(directory+"train-images-idx3-ubyte")
    y_train = idx2numpy.convert_from_file(directory+"train-labels-idx1-ubyte")

x_test = x_test.reshape(10000,784)
x_train = x_train.reshape(60000,784)

#Hog Feature Extraction for training set
list_hog_train = []
for feature in x_train:
    fd = hog(feature.reshape((28,28)), orientations=10, pixels_per_cell=(7,7),cells_per_block=(1,1),visualise=False )
    list_hog_train.append(fd)
hog_features = np.array(list_hog_train, 'float64')
preProcess = preprocessing.MaxAbsScaler().fit(hog_features)
hog_features_transformed_train = preProcess.transform(hog_features)
print(hog_features_transformed_train.shape)

#Hog featuure extraction for testing set
list_hog_test = []
for feature in x_test:
    fd = hog(feature.reshape((28,28)), orientations=10, pixels_per_cell=(7,7),cells_per_block=(1,1),visualise=False )
    list_hog_test.append(fd)
hog_features_test = np.array(list_hog_test, 'float64')
preProcess = preprocessing.MaxAbsScaler().fit(hog_features_test)
hog_features_transformed_test = preProcess.transform(hog_features_test)
print(hog_features_transformed_test.shape)

model = SVC()
model.fit(hog_features_transformed_train,y_train)
y_pred = model.predict(hog_features_transformed_test)

for i in y_pred:
    print(i)

# print(accuracy_score(y_test, y_pred))

