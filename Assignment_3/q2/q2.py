url_train_csv = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_3/q2/sample_train_2.txt'
url_test_csv = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_3/q2/sample_test_2.txt'
output_sample_csv = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_3/q2/output_of_sample_test_2.txt'
dataset_url = '/home/abhishek/dev/Semester_2/SMAI/Assignments/Assignment_3/q2/dataset'

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import transform,io
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def create_train(self):
        dir_list = os.listdir(dataset_url)
        f = open("sample_train_2.txt",'w')
        for i in dir_list:
            x = i.split("_")
            f.write("./dataset/"+i+" "+x[0]+"\n")
        f.close()

    def create_test(self):
        f = open("sample_test_2.txt",'w')
        f1 = open("output_of_sample_test_2.txt",'w')
        dir_list = os.listdir(dataset_url)
        for i in range(0,len(dir_list)):
            if(i%3==0):
                pass
            x = dir_list[i].split("_")
            f.write("./dataset/"+dir_list[i]+"\n")
            f1.write(x[0]+"\n")
        f.close()
        f1.close()

    def read_csv_file(self,marker):
        image_files = []            
        f = None
        if(marker == "train"):
            f = open(url_train_csv,'r')
        elif(marker == "test"):
            f = open(url_test_csv,'r')
        else:
            f = open(output_sample_csv,'r')

        lines = f.readlines()
        for line in lines:
            image_files.append(line.strip()) 
        return image_files

    def read_sample_output(self):
        x=0
    
    def split_image_dir_and_label(self,images_directory,marker):
        if(marker == "train"):
            # print("splitting directory and labels")
            label = []
            directory = []
            for l in images_directory:
                x = l.split(" ")
                label.append(x[1])
                directory.append(x[0])
            # print("splitting directory and labels done ....")
            return label, directory
        elif(marker == "test"):
            # print("read lines...")
            lines = []
            for l in images_directory:
                lines.append(l)
            # print("reading lines done...")
            return lines
            
    def read_image_from_directory_to_grayscale(self,directory):
        # print("reading images from directory and downscaling images")
        faces = []
        for i in directory:
            img = io.imread(i)
            img = img.astype(np.uint8)
            # converting to grayscale
            rgb_weights = [0.2989, 0.5870, 0.1141]
            grayscale_image = np.dot(img[...,:3], rgb_weights)
            small_grey = transform.resize(grayscale_image, (64,64), mode='symmetric', preserve_range=True)
            reshape_img = small_grey.reshape(1, 4096)
            faces.append(reshape_img[0])
        X = np.asarray(faces)
        # print("reading images from directory and downscaling images done...")
        return X
    
    def apply_pca(self,X):
        # print("Applying PCA on the image set")
        eig_val, eig_mat = np.linalg.eig(np.cov(X))
        idx = eig_val.argsort()[::-1]
        eig_val = eig_val[idx]
        eig_mat = eig_mat[:,idx]
        eigen_coeff = eig_mat[:,range(50)]
        X_PCA = np.dot(eigen_coeff.T,X)
        # print("Applying PCA on the image set done .......")
        return X_PCA.T
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def cost(self,w,b,X,y,lmd=10):
        # print("Calculating Cost")
        m = X.shape[0]
        z = np.matmul(X,w) + b
        hx = self.sigmoid(z)
        J = (-1/m)*( np.sum( y * np.log(hx) + (1. - y) * np.log(1.- hx) ) )
        J += (lmd/(2*m))* np.matmul(w,w)
        # print("Calculating Cost done ..")
        return J

    def gradient_descent(self,w,b,X,y,learning_rate=0.01,lmd=10,no_of_iteration=10000):
        # print("Applying Gradient Descent")
        m = X.shape[0]
        # print("Initial cost: {}".format( self.cost(w,b,X,y) ))
        for i in range(no_of_iteration):
            z = np.matmul(X,w) + b
            hx = self.sigmoid(z)
            dw = (1/m)*np.matmul(X.T,hx-y)
            db = (1/m)*np.sum(hx-y)
            factor = 1-( (learning_rate * lmd)/m)
            w = w*factor - learning_rate*dw
            b = b - learning_rate*db
            # if i % 500 == 0:
                # print("Iteration {} cost :{}".format(i,self.cost(w,b,X,y)))
        # print("Final cost: {}".format( self.cost(w,b,X,y) ))
        # print("Applying Gradient Descent done ....")
        return w,b     

    def accuracy(self,w,b,X,y):
        # print("Calculating Accuracy")
        m = X.shape[0]
        z = np.matmul(X,w) + b
        hx = self.sigmoid(z)
        pred = np.round(hx)
        correct_pred = (pred==y)
        total = np.sum(correct_pred)
        # print("Calculating Accurcy done .....")
        return (total*100)/m     

    def test(self,w,b,X):
        m = X.shape[0]
        z = np.matmul(X,w)+b 
        hx = self.sigmoid(z)
        pred = np.round(hx)
        return pred
 

lr = LogisticRegression()
# Creating Train File
# lr.create_train()
# lr.create_test()

lines = lr.read_csv_file("train")
# print(images_directory)
label, directory = lr.split_image_dir_and_label(lines,"train")
# label = np.asarray(label)
# print(label)
unique_labels = np.unique(np.asarray(label))
# print(unique_labels)
X_train = lr.read_image_from_directory_to_grayscale(directory)
# normalizing
X_train = X_train/255
#applying PCA
X_PCA = lr.apply_pca(X_train.T)
print(X_PCA.shape)
# print(X_PCA[0])
m = X_train.shape[0]
weight_label_map = {}
i = 0
THETA = []
BIAS = []

for unique in unique_labels:
    y_train = []
    for l in label:
        if(unique == l):
            y_train.append(1)
        else:
            y_train.append(0)
    
    weight_label_map[i] = unique
    i = i+1
    y_train = np.asarray(y_train)
    # print(y_train)
    w = np.zeros(X_PCA.shape[1],dtype=np.float64)
    b = 0.0
    w,b = lr.gradient_descent(w,b,X_PCA,y_train)
    THETA.append(w)
    BIAS.append(b)

print(weight_label_map)

THETA = np.asarray(THETA)
BIAS = np.asarray(BIAS)

lr1 = LogisticRegression()
lines = lr1.read_csv_file("test")
directory = lr1.split_image_dir_and_label(lines,"test")
label_test = lr1.read_csv_file("label")
labels = lr1.split_image_dir_and_label(label_test,"test")
X_test = lr1.read_image_from_directory_to_grayscale(directory)
X_test = X_test/255
X_Test_PCA = lr1.apply_pca(X_test.T)
# print(X_Test_PCA.shape)

y_pred = []
for X in X_Test_PCA:
    i = 0
    max_hx = 0.0
    index = 0
    for w,b in zip(THETA,BIAS):
        pred = lr1.test(w,b,X)
        if(pred > max_hx):
            max_hx = pred
            index = i
        i = i+1
    y_pred.append(weight_label_map[index])

# print(y_pred)
# print(label_test)
y_pred = np.asarray(y_pred)
print(accuracy_score(label_test, y_pred))