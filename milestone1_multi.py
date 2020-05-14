import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from Pre_processing import *

data = pd.read_csv('Hotel_Reviews.csv')

data.dropna(how='any',inplace=True)
hotel_data=data.iloc[:,:]
X=data.iloc[:,1:17] #Features
Y=data['Reviewer_Score'] #Label
cols=('Review_Date','Hotel_Name','Reviewer_Nationality','Negative_Review','Positive_Review')



X=Feature_Encoder(X,cols);

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30,shuffle=True)

corr = hotel_data.corr()

top_feature = corr.index[abs(corr['Reviewer_Score']>0.5)]

plt.subplots(figsize=(12, 8))
top_corr = hotel_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

cls = linear_model.LinearRegression()

cls.fit(X_train,y_train)
prediction= cls.predict(X_test)

print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(y_test), prediction))

true_player_value=np.asarray(y_test)[0]
predicted_player_value=prediction[0]

print('True value for the first player in the test set in millions is : ' + str(true_player_value))
print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))