from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import math

train_url = 'https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q3/train.csv'
test_url = 'https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q3/test.csv'
test_label_url = 'https://raw.githubusercontent.com/abhishek0508/Machine-Learning/master/Assignment_1/OneDrive_1_21-01-2020/q3/test_labels.csv'

class Node:
    # split_val=0
    # col=''
    # price=0
    # isleaf=0
    # df = None
    # def __init__(self, df,col,split_val,isleaf):
    #     self.left = None
    #     self.right = None
    #     self.df = df
    #     self.split_val=split_val
    #     self.col=col
    #     self.isleaf=isleaf
    
    # def __init__(self, df):
    #     self.df = df


class Tree:
    MAX= 1e20
    cnt=0
    # def makeTree(self,root,df):
    #     print("Here 1")
    #     sub_df=root.df
    #     root.price=sub_df["SalePrice"].mean()
        
    #     if sub_df.shape[0]<30:
    #         isleaf=1
    #         return root
        
    #     col,val=self.get_best_feature(sub_df)
    #     root.col=col
    #     root.split_val=val
    #     if col == '':
    #         print('None wala',col)
    #         return None
        
    #     print("Here 2")
    #     isleaf=0
    #     df.reset_index(drop=True)
    #     print("Here 3")

    #     left_df = sub_df.loc[df[col] <= val] 
    #     right_df = sub_df.loc[df[col] > val]

    #     left_df = left_df.drop(col, 1)
    #     right_df = right_df.drop(col, 1)
    #     #print('col:',root.col,'val: ',root.split_val,' left:',left_df.shape,' right',right_df.shape)
    #     #print('col',root.col)
    #     print("Here 4")
    #     root.left=self.makeTree(Node(left_df),df)
    #     root.right=self.makeTree(Node(right_df),df)
    #     # global cnt
    #     # cnt=cnt+1
    #     return root
        
    # def split_error(self,vecx,vecy,valx):
    #     left=[]
    #     right=[]
    #     for index in range(len(vecx)):
            
    #         if vecx[index] <= valx:
    #             left.append(vecy[index])
    #         else:
    #             right.append(vecy[index])

    #     if len(left)==0:
    #         return MAX
    #     if len(right)==0:
    #         return MAX
            
    #     mean_left=np.mean(left)
    #     mean_right=np.mean(right)
    #     left=np.asarray(left)
    #     right=np.asarray(right)
    #     A=[mean_left] * len(left)
    #     B=[mean_right] * len(right)


    #     mse_l=np.square(np.subtract(left,A)).mean()
    #     mse_r=np.square(np.subtract(right,B)).mean()
    #     mse=mse_l*len(left) + mse_r*len(right)
    #     mse=mse/len(vecy)
    #     return mse

    
 
    # def get_min_split(self,vecx,vecy):
    #     sorted_vecx=np.unique(np.sort(vecx))
    #     min_mse=self.MAX
    #     min_valx=0
    #     for i in range(1,len(sorted_vecx)):
    #         valx=(sorted_vecx[i]+sorted_vecx[i-1])/2
    #         x=self.split_error(vecx,vecy,valx)
    #         if x < min_mse:
    #             min_mse=x
    #         min_valx=valx
        
    #     return min_mse,min_valx


    # def get_best_feature(self,df):
    #     sales_prices=df['SalePrice'].to_numpy()
    #     min_mse=self.MAX
    #     min_col=""
    #     min_valx=0
    #     for col in df.columns:
            
    #         if(col=='SalePrice'):
    #             continue  
    #         feature=df[col].to_numpy()
    #         mse,valx=self.get_min_split(feature,sales_prices)
    #         if mse < min_mse:
    #             min_mse=mse
    #         min_col=col
    #         min_valx=valx
    #     if(min_col==''):
    #         print('col',min_col)  
    #     return min_col,min_valx

 
    # def traverse(self,node,vec_df):
    #     col=node.col
    #     print(node.col)
    #     if node.isleaf:
    #         #print("Found:")  
    #         return node.price

    #     if col=='':
    #         return node.price

    #     val=vec_df[col]
            
    #     if val <= node.split_val:
    #         if node.left is None:
    #             return node.price
    #         #print("Going Left")
    #         else:
    #             return self.traverse(node.left,vec_df)

    #     elif val > node.split_val:
    #         if node.right is None:
    #             return node.price
    #         else:
    #         #print("Going Right")
    #             return self.traverse(node.right,vec_df)


class DecisionTree:
    # print("Start Train")
    # tree = Tree()
    # df_dummy = None
    # root = None

    # def train(self,url):
    #     pd.set_option('display.max_rows', None)
    #     pd.set_option('display.max_columns', None)
    #     train_df = pd.read_csv(url)

    #     # print(train_df)
    #     # print("Here 1")
    #     missing_values={'Alley':'NA','BsmtQual':'NA','BsmtCond':'NA','BsmtExposure':'NA',
    #             'BsmtFinType1':'NA','BsmtFinType2':'NA','FireplaceQu':'NA','GarageQual':'NA',
    #             'GarageCond':'NA','PoolQC':'NA','Fence':'NA','MiscFeature':'NA','GarageType':'NA',
    #             'GarageFinish':'NA','GarageYrBlt':'NA','MasVnrType':'NA','MasVnrArea':0,
    #             'Electrical':train_df['Electrical'].mode()[0],'LotFrontage':int(train_df['LotFrontage'].mean())}

    #     df=train_df.fillna(value=missing_values)

    #     # print("Here 2")
    #     cols=df.columns
    #     num_cols = df._get_numeric_data().columns
    #     categorical_cols = list(set(cols) - set(num_cols))
    #     categorical_cols.remove('GarageYrBlt')
    #     cat_dummy_df=pd.get_dummies(df[categorical_cols])
    #     df.drop(columns=categorical_cols,inplace=True)
    #     df=pd.concat([df,cat_dummy_df],axis=1)
    #     # print("Here 3")

    #     df.drop(columns=['GarageYrBlt','Id'],inplace=True) 

    #     print(df)
    #     self.root = Node(df)
    #     self.tree.makeTree(self.root,df)


    # def predict(self,url):
    #     print("here 5")
    #     test_df= pd.read_csv(url)
    #     missing_values={'Alley':'NA','BsmtQual':'NA','BsmtCond':'NA','BsmtExposure':'NA',
    #         'BsmtFinType1':'NA','BsmtFinType2':'NA','FireplaceQu':'NA','GarageQual':'NA',
    #         'GarageCond':'NA','PoolQC':'NA','Fence':'NA','MiscFeature':'NA','GarageType':'NA',
    #         'GarageFinish':'NA','GarageYrBlt':'NA','MasVnrType':'NA','MasVnrArea':0,
    #         'Electrical':test_df['Electrical'].mode()[0],'LotFrontage':int(test_df['LotFrontage'].mean())}

    #     df1=test_df.fillna(value=missing_values)
    
    #     print("Here 6")

    #     cols=df1.columns
    #     num_cols = df1._get_numeric_data().columns
    #     categorical_cols = list(set(cols) - set(num_cols))
    #     categorical_cols.remove('GarageYrBlt')
    #     cat_dummy_df1=pd.get_dummies(df1[categorical_cols])
    #     df1.drop(columns=categorical_cols,inplace=True)
    #     df1=pd.concat([df1,cat_dummy_df1],axis=1)

    #     df1.drop(columns=['GarageYrBlt','Id'],inplace=True)

    #     df1 = df1.fillna(0)
        
    #     print("#####################Test Data###################")
    #     print(df1)

    #     print("Here 7")

    #     pred_prices = []
    #     for i in range (1, df1.shape[0]):
    #         pred_prices.append(self.tree.traverse(self.root,df1.iloc[i]))
    #         # print(pred_prices)
       
    #     pred_prices.append(100000)
    #     print(len(pred_prices))
    #     return pred_prices


##############code to run in google colab####################
