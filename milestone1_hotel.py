import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Pre_processing import *
from sklearn.preprocessing import PolynomialFeatures


needed=["Average_Score","Review_Total_Negative_Word_Counts","Review_Total_Positive_Word_Counts",'Reviewer_Score']

def drop_columns(data, needed):
    not_needed_columns = [c for c in data if c not in needed]
    data.drop(not_needed_columns, axis=1, inplace=True)
    return data

Missing_Val =["NA",' ','']

data = pd.read_csv('Hotel_Reviews.csv',na_values=Missing_Val)
data1=pd.read_csv('hotel_reviews_regression_test.csv',na_values=Missing_Val)
#data.dropna(axis=0,how='any',thresh=None,subset=["lat","lng"],inplace=True)

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

mean_lat = data['lat'].mean()
mean_lng = data['lng'].mean()
mean_Additinal_num_Score = data['Additional_Number_of_Scoring'].mean()
mean_Av_Score = data['Average_Score'].mean()
mean_Neg_Count = data['Review_Total_Negative_Word_Counts'].mean()
mean_Numb_Rev = data['Total_Number_of_Reviews'].mean()
mean_Numb_Reviwer_Revs = data['Total_Number_of_Reviews_Reviewer_Has_Given'].mean()
mean_Pos_Count = data['Review_Total_Positive_Word_Counts'].mean()

data1['lat'].fillna(mean_lat, inplace=True)
data1['lng'].fillna(mean_lng, inplace=True)
data1['Additional_Number_of_Scoring'].fillna(mean_Additinal_num_Score, inplace=True)
data1['Average_Score'].fillna(mean_Av_Score, inplace=True)
data1['Review_Total_Negative_Word_Counts'].fillna(mean_Neg_Count, inplace=True)
data1['Total_Number_of_Reviews'].fillna(mean_Numb_Rev, inplace=True)
data1['Total_Number_of_Reviews_Reviewer_Has_Given'].fillna(mean_Numb_Reviwer_Revs, inplace=True)
data1['Review_Total_Positive_Word_Counts'].fillna(mean_Pos_Count, inplace=True)

data1['Negative_Review'].fillna('No Negative',inplace=True)
data1['Positive_Review'].fillna('No Positive',inplace=True)
data1['Hotel_Address'].fillna('No Hotel',inplace=True)
data1['Review_Date'].fillna('No Date',inplace=True)
data1['Hotel_Name'].fillna('No Name',inplace=True)
data1['Reviewer_Nationality'].fillna('No Nationality',inplace=True)
data1['Tags'].fillna('No Tags',inplace=True)
data1['days_since_review'].fillna('No Days',inplace=True)

mean_lat = data1['lat'].mean()
mean_lng = data1['lng'].mean()
mean_Additinal_num_Score = data1['Additional_Number_of_Scoring'].mean()
mean_Av_Score = data1['Average_Score'].mean()
mean_Neg_Count = data1['Review_Total_Negative_Word_Counts'].mean()
mean_Numb_Rev = data1['Total_Number_of_Reviews'].mean()
mean_Numb_Reviwer_Revs = data1['Total_Number_of_Reviews_Reviewer_Has_Given'].mean()
mean_Pos_Count = data1['Review_Total_Positive_Word_Counts'].mean()


cols=('Hotel_Address','Review_Date','Hotel_Name','Reviewer_Nationality')

data=Feature_Encoder(data,cols);
data1=Feature_Encoder(data1,cols)

'''
corr = data.corr()
plt.subplots(figsize=(12, 8))
sns.heatmap(corr, annot=True)
plt.show()
'''
data = drop_columns(data, needed)
data1= drop_columns(data1,needed)

X = np.array(data.iloc[:, :len(data.columns) - 1])
Y = np.array(data["Reviewer_Score"])
W=np.array(data1.iloc[:, :len(data1.columns)-1])
Z= np.array(data1["Reviewer_Score"])
def normalize_data_min_max(x_features):
    # loop on Each Column (Features in X)
    for i in range(x_features.shape[1]):
        x_features[:, i] = (x_features[:, i] - min(x_features[:, i])) / (max(x_features[:, i]) - min(x_features[:, i]))
    return x_features


X = normalize_data_min_max(X)
W=normalize_data_min_max(W)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)


cls = linear_model.LinearRegression()

cls.fit(X_train, y_train)
prediction = cls.predict(W)


print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Z), prediction))


true_score_value=np.asarray(Z)[0]
predicted_score_value=prediction[0]

print('true score value for multiple_linear regression : ' + str(true_score_value))
print('Predicted score value for multiple_linear regression : ' + str(predicted_score_value))

n_degree = int(input("Please Enter the Degree Of the poly : "))
poly_feature = PolynomialFeatures(degree=n_degree)
x_poly_train = poly_feature.fit_transform(X_train)
x_poly_test = poly_feature.fit_transform(X_test)

W_poly_test = poly_feature.fit_transform(W)


print("poly Regression data : ")
cls = linear_model.LinearRegression()
cls.fit(x_poly_train, y_train)
prediction = cls.predict(W_poly_test)

print("The Mean Square Error is : ", metrics.mean_squared_error(Z, prediction))
print("The coefficients are ", cls.coef_)
print("The Intercept is : ", cls.intercept_)

true_score_value=np.asarray(y_test)[0]
predicted_score_value=prediction[0]

print('true score value for polynomial regression : ' + str(true_score_value))
print('Predicted score value for polynomial regression : ' + str(predicted_score_value))
