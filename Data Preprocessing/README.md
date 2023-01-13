# Data Preprocessing

## Pre-Processing Steps 

1. Import relevant libraries 
2. Import the Data Set to analyse and clean for prediction 
3. Missing value treatment(Impute the dataset)
- In numerical data
Populate with mean –When there  is no outlier
Populate with median – When there is outlier
- Categorical data– Mode imputation – Maximum data category


## 1. Import Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
```

## 2. Load Data Set
### Loading xlsx files 
``` 
os.chdir ("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df1= pd.read_excel("Data Preprocessing Data File.xlsx")	
print (df1)
```

### Loading csv files 
``` 
os.chdir ("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df1= pd.read_csv("Data Preprocessing Data File.csv")
print (df1)
```


## 3. Load independent variables and dependent variables to two separate arrays 
### Columns Independent variable – Country, Age , Salary (Create X)
```
x = df1.iloc[:,:-1].values
print (x)
```

### Dependent variable  - Purchased (Create Y )
```
y = df1.iloc[:,-1].values
print (y)
```

## Encoding
1. Label  Encoding – Convert to number 
2. One hot Encoding – Create multiple columns based on number of unique record count
3. Feature Scaling 
4. Standardization – When there  no outlier 
5. Normalization/ Min Max Scaler :  When there is  outlier




