import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Label_Endcoder import *
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import r2_score
from timeit import default_timer as timer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

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

corr = data.corr()
plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.show()
needed=['Reviewer_Score','Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality','Tags','Review_Total_Positive_Word_Counts','Review_Total_Negative_Word_Counts']

def drop_columns(data, needed):
    not_needed_columns = [c for c in data if c not in needed]
    data.drop(not_needed_columns, axis=1, inplace=True)
    return data

data = drop_columns(data, needed)
X = np.array(data.iloc[:, :len(data.columns) - 1])
Y = np.array(data["Reviewer_Score"][:])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1))
start=timer()
svm_model_linear_ovr.fit(X_train, y_train)
end=timer()
train_time_ovr=end-start
#svm_predictions = svm_model_linear_ovr.predict(X_test)
start=timer()
ovr_prediction=svm_model_linear_ovr.predict(X_test)
end=timer()
test_time_ovr=end-start
# model accuracy for X_test
accuracy = svm_model_linear_ovr.score(X_test, y_test)
print('One VS Rest SVM accuracy: ' + str(accuracy))
print('Mean Square Error for OVR SVM', metrics.mean_squared_error(np.asarray(y_test), ovr_prediction))
print('r2 score OVR is: '+ str(r2_score(y_test, ovr_prediction)))
print('train time OVR is :'+ str(train_time_ovr))
print('test time OVR is :'+ str(test_time_ovr))

print('--------------------------------------------------------------------------------')
#
svm_model_linear_ovo = SVC(kernel='linear', C=1)
start=timer()
svm_model_linear_ovo.fit(X_train, y_train)
end=timer()
ovo_train_time=end-start
#svm_predictions = svm_model_linear_ovo.predict(X_test)
start=timer()
ovo_prediction = svm_model_linear_ovo.predict(X_test)
end=timer()
ovo_test_time=end-start
# model accuracy for X_test
accuracy = svm_model_linear_ovo.score(X_test, y_test)
print('One VS One SVM accuracy: ' + str(accuracy))
print('Mean Square Error for OVO SVM', metrics.mean_squared_error(np.asarray(y_test), ovo_prediction))
print('r2 score OVO is: '+ str(r2_score(y_test, ovo_prediction)))
print('train time OVO is :'+ str(ovo_train_time))
print('test time OVO is :'+ str(ovo_test_time))
