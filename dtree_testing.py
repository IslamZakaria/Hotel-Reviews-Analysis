import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Label_Endcoder import *
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from sklearn.decomposition import PCA
import pickle


Missing_Val =["NA",' ','']
data = pd.read_csv('Hotel_Review_Milestone_2_Test_Samples.csv',na_values=Missing_Val)

mean_lat = data['lat'].mean()
mean_lng = data['lng'].mean()
mean_Additinal_num_Score = data['Additional_Number_of_Scoring'].mean()
mean_Av_Score = data['Average_Score'].mean()
mean_Neg_Count = data['Review_Total_Negative_Word_Counts'].mean()
mean_Numb_Rev = data['Total_Number_of_Reviews'].mean()
mean_Numb_Reviwer_Revs = data['Total_Number_of_Reviews_Reviewer_Has_Given'].mean()
mean_Pos_Count = data['Review_Total_Positive_Word_Counts'].mean()

data['lat'].fillna(mean_lat, inplace=True)
data['lng'].fillna(mean_lng, inplace=True)
data['Additional_Number_of_Scoring'].fillna(mean_Additinal_num_Score, inplace=True)
data['Average_Score'].fillna(mean_Av_Score, inplace=True)
data['Review_Total_Negative_Word_Counts'].fillna(mean_Neg_Count, inplace=True)
data['Total_Number_of_Reviews'].fillna(mean_Numb_Rev, inplace=True)
data['Total_Number_of_Reviews_Reviewer_Has_Given'].fillna(mean_Numb_Reviwer_Revs, inplace=True)
data['Review_Total_Positive_Word_Counts'].fillna(mean_Pos_Count, inplace=True)

data['Negative_Review'].fillna('No Negative',inplace=True)
data['Positive_Review'].fillna('No Positive',inplace=True)
data['Hotel_Address'].fillna('No Hotel',inplace=True)
data['Review_Date'].fillna('No Date',inplace=True)
data['Hotel_Name'].fillna('No Name',inplace=True)
data['Reviewer_Nationality'].fillna('No Nationality',inplace=True)
data['Tags'].fillna('No Tags',inplace=True)
data['days_since_review'].fillna('No Days',inplace=True)

#print(data["Negative_Review"][286])
cols=('Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Reviewer_Score','Tags')

data=Feature_Encoder(data,cols);

needed=['Reviewer_Score','Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Tags','Review_Total_Positive_Word_Counts','Review_Total_Negative_Word_Counts']
def drop_columns(data, needed):
    not_needed_columns = [c for c in data if c not in needed]
    data.drop(not_needed_columns, axis=1, inplace=True)
    return data

data = drop_columns(data, needed)
X = np.array(data.iloc[:, :len(data.columns) - 1])
Y = np.array(data["Reviewer_Score"])

scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X)

'''
pca = PCA(.85)
pca.fit(X)
X_train = pca.transform(X)
'''

dtree_train=pickle.load(open("dtree",'rb'))

test=dtree_train.predict(X)
accuracy=np.mean(test == Y)*100

print('accuracy :'+ str(accuracy))


