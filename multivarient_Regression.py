import pandas as pd
import numpy as np
import matplotlib as mlp
from sklearn import linear_model
import math

data = pd.read_csv('homeprices.csv')

#Handling the missing vlaues


median_rooms = math.floor(data.bedrooms.median())

#preprocessing data
#missing the values is replaced with median of the column 
data.bedrooms = data.bedrooms.fillna(median_rooms)
new_data =data.to_csv('preprocessed_data.csv')
modified =pd.read_csv('preprocessed_data.csv')
mulvalreg = linear_model.LinearRegression()
mulvalreg.fit(modified[['area','bedrooms','age']],modified.price)

input_data = pd.DataFrame([[2600,3.0,20]], columns=['area', 'bedrooms', 'age'])
prediction = mulvalreg.predict(input_data)[0]  # Extracting the single predicted value
prediction_rounded = math.floor(prediction)

print("Predicted Price:", prediction_rounded)