# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['type'] = iris.target

class MyKNN:
    def __init__(self, k=5):
        self.k = k
    
#1. Convert and assign features and target data to variable
    def my_fit(self, input_data, target):
        self.feature_data = input_data
        self.target_data = target

#2. Calculate Euclidean Distance
    def calculate_distance_vector_matrix(self, one_data):
        """
        one_data = green point (image example)
        """
        distances = np.sqrt(np.sum(np.square(self.feature_data - one_data), axis = 1))
        return distances
#3. Sort the distance 
    def find_k_neighbors(self, one_data):
        closDist = self.calculate_distance_vector_matrix(one_data)
        return closDist.argsort()[:self.k]

#4. Find the index of the closest- return category
    def find_category(self, one_data):
        indexNeigh = self.find_k_neighbors(one_data)
        return self.target_data[indexNeigh]
    
#5. Finding the Majority
    def my_predict(self, one_data):
        classes = self.find_category(one_data)
        return np.bincount(classes).argmax()
  
model = MyKNN(k=7)    
feature_data = df.drop(columns=['type'], axis = 1)
target_data = df.type

model.my_fit(feature_data, target_data)
one_data = [5.1, 3.5, 1.5, 0.1]
my_pred = model.my_predict(one_data)

print(my_pred)



#######Appendix
a = np.array([3, 8, 1, 4, 9])  # 1, 3, 4, 8, 9
print(a.argsort()[:3])

b = np.array([0,0,1,0,1,1,1])
print(np.bincount(b))
print(np.bincount(b).argmax())



