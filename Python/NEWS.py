##pip install -U scikit-learn

import numpy as np 
import pandas as pd
import itertools



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


#Read the data
df=pd.read_csv('C:/Users/pavan/Downloads/organizations-100/organizations-100.csv')


#Get shape and head
df.shape
cat = df.head()

print(cat)
