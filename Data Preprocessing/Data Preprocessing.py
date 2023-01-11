# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2. Load Data Set 
os.chdir ("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df1= pd.read_excel("Data Preprocessing Data File.xlsx")	
print (df1)

# 3. Load independent variables and dependent variables to two separate arrays 
# Columns Independent variable – Country, Age , Salary 
# Dependent variable  - Purchased

# Create X
x = df1.iloc[:,:-1].values
print (x)


# Create Y 
y = df1.iloc[:,-1].values
print (y)

# 4. Missing value treatment – Impute Values 
# Country – Most Frequent 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(x[:,0:1])
x [:,0:1]= imputer.transform(x[:,0:1])
print (x)

# Missing value treatment – Impute Values 
# Age – Constant 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=40)
imputer = imputer.fit(x[:,1:2])
x [:,1:2]= imputer.transform(x[:,1:2])
print (x)


# Missing value treatment – Impute Values 
# Salary – Mean 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(x[:,2:3])
x [:,2:3]= imputer.transform(x[:,2:3])
print (x)

# 5. Label Encoding 
from sklearn.preprocessing import LabelEncoder
label_x = LabelEncoder()
x[:,0]= label_x.fit_transform(x[:,0])
print (x)

# One hot encoding /Column Transformation 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print (x)

# Label Encoding Y
label_y = LabelEncoder()
y = label_y.fit_transform(y)
print (y)

# Standardization
from sklearn.preprocessing import StandardScaler
std_sca= StandardScaler()
x_STD = std_sca.fit_transform(x)
print (pd.DataFrame(x_STD))

# Normalization - MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
Nm_x= MinMaxScaler()
x_NOR = Nm_x.fit_transform(x)
print (pd.DataFrame(x_NOR))

# Normalization 
from sklearn.preprocessing import Normalizer
Nm_x= Normalizer()
x_NOR = Nm_x.fit_transform(x)
print (pd.DataFrame(x_NOR))

# Model Creation 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split (x_NOR,y,test_size = 0.2)
print (pd.DataFrame(x_train), y_train)






