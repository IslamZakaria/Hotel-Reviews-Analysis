import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from Label_Endcoder import *
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from sklearn.decomposition import PCA
import pickle
from try_Encode import *

Missing_Val =["NA",' ','']
#data = pd.read_csv('Hotel_Reviews_Milestone_2.csv',na_values=Missing_Val)
data = pd.read_csv('hotel_reviews_classification_test_shuffled.csv',na_values=Missing_Val)
print(data.__len__())
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
cols=('Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Reviewer_Score')

data=Feature_Encoder(data,cols);

'''
corr = data.corr()
plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.show()
'''
needed=['Reviewer_Score','Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Review_Total_Positive_Word_Counts','Review_Total_Negative_Word_Counts']

#needed=['Reviewer_Score','Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Review_Total_Positive_Word_Counts','Review_Total_Negative_Word_Counts','lat','lng','Additional_Number_of_Scoring','Average_Score','Total_Number_of_Reviews_Reviewer_Has_Given','Tags','Total_Number_of_Reviews','days_since_review']


def drop_columns(data, needed):
    not_needed_columns = [c for c in data if c not in needed]
    data.drop(not_needed_columns, axis=1, inplace=True)
    return data

data = drop_columns(data, needed)
X = np.array(data.iloc[:414739, :len(data.columns) - 1])
Y = np.array(data["Reviewer_Score"][:414739])

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
#X_test = scaler.transform(X_test)

error = []
accuracy=[]
# Calculating error for K values between 1 and 40

'''
pca = PCA(n_components=10)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''


'''
knn = KNeighborsClassifier(n_neighbors=35)
start = timer()
knn.fit(X_train, y_train)
end = timer()
train_time = end - start
pickle.dump(knn, open('knn', 'wb'))
'''


trained_model=pickle.load(open("knn",'rb'))
test=trained_model.predict(np.array(X))
accuracy=np.mean(test==Y)*100
print('accuracy :'+ str(accuracy))
true_score_value = np.asarray(Y)[0]
predicted_score_value = test[0]
print('true score value  : ' + str(true_score_value))
print('Predicted score value : ' + str(predicted_score_value))


'''
start = timer()
pred_i = knn.predict(X_test)
end = timer()
test_time = end - start

error.append(np.mean(pred_i != y_test))
accuracy.append(np.mean(pred_i == y_test) * 100)
true_score_value = np.asarray(y_test)[0]
predicted_score_value = pred_i[0]

print('i = :' + str(35))
print('Mean Square Error for KNN', metrics.mean_squared_error(np.asarray(y_test), pred_i))
print("The achieved accuracy using KNN is " + str(accuracy))
print('true score value for KNN : ' + str(true_score_value))
print('Predicted score value KNN : ' + str(predicted_score_value))
print('r2 score is: ' + str(r2_score(y_test, pred_i)))
print('train time is :' + str(train_time))
print('test time is :' + str(test_time))
'''

'''
print('----------------------------------------------------------------')


knn = KNeighborsClassifier(n_neighbors=3)
start=timer()
knn.fit(X_train, y_train)
end=timer()
train_time=end-start

start=timer()
pred_i = knn.predict(X_test)
end=timer()
test_time=end-start
accuracy=np.mean(pred_i == y_test)*100

true_score_value=np.asarray(y_test)[0]
predicted_score_value=pred_i[0]

print('Mean Square Error for KNN', metrics.mean_squared_error(np.asarray(y_test), pred_i))
print("The achieved accuracy using KNN is " + str(accuracy))
print('true score value for KNN : ' + str(true_score_value))
print('Predicted score value KNN : ' + str(predicted_score_value))
print('r2 score is: '+ str(r2_score(y_test, pred_i)))
print('train time is :'+ str(train_time))
print('test time is :'+ str(test_time))



plt.figure(figsize=(12, 6))
plt.plot(i, error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(i, accuracy, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Acc')
plt.show()

'''