# Data Preprocessing

## Pre-Processing Steps 

1. Import relevant libraries 
2. Import the Data Set to analyse and clean for prediction 
3. Missing value treatment(Impute the dataset)
- In numerical data
Populate with mean –When there  is no outlier
Populate with median – When there is outlier
- Categorical data– Mode imputation – Maximum data category

## 1 Reading the data

### 1.1 Import Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import os
```

### 1.2 Load Data Set
#### 1.2.1 Loading xlsx files 
``` 
os.chdir("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df = pd.read_excel("Data Preprocessing Data File.xlsx")	
print(df)
```

#### 1.2.2 Loading csv files 
``` 
os.chdir("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df1 = pd.read_csv("Data Preprocessing Data File.csv")
print(df)
```

### 1.3 Identifying the dimensions

#### 1.3.1 Shape of the dataset
```
df.shape
```

#### 1.3.2 Columns in the dataset
```
df.columns
```

#### 1.3.3 Top values of the dataset
```
df.head()
```

## 2 Variable identification

#### 2.1 Identifying the datatypes of the variables
```
df.dtypes
```

## 3 Univarient Analysis

### 3.1 Contineous Variable

#### 3.1.1 Tabular Methods
```
df.describe()
```

#### 3.1.2 Graphical method
##### 3.1.2.1 Plotting histogram
```
df['variable'].plot.hist
```

##### 3.1.2.2 Plotting boxplot
```
df['variable'].plot.box()
```

## 3.2 Categorical Variable

### 3.2.1 Tabular Methods
#### 3.2.1.1 Creating frequency table
```
df['variable'].value_counts() 
```

#### 3.2.1.2 Create percentages frequencies 
```
df['variable'].value_counts()/len(df['variable']) 
```
### 3.2.2 Graphical method
#### 3.2.2.1 Plotting barplot 
```
df['variable'].value_counts().plot.bar() 
```

#### 3.2.2.2 Plotting barplot using percentage 
```
(df['variable'].value_counts()/len(df['variable'])).plot.bar() 
```


## 4 Bivarient Analysis

### 4.1 Contineous - Contineous Variable

#### 4.1.1 Plotting scatter plot 
```
df.plot.scatter('variable 1','variable 2')
```

#### 4.1.2 Using correlation matrix
```
df.corr()
``` 

### 4.2 Categorical - Contineous Variable
#### 4.2.1 Using tabular method
```
df.groupby('Categorical variable')['contineous variable'].mean()
```

#### 4.2.2 Creating bar plot
```
df.groupby('Categorical-variable')['contineous-variable'].mean().plot.bar
```

#### 4.2.3 By using t-test  
Create groups of categorical variables
```
group = df[df['categorical-variable']=='Value']
```
Calculating t-test for each categorical variable
```
ttest_ind[group1['contineous-variable'],,nan-policy='omit']
```

### 4.3 Categorical - Categorical Variable
#### 4.3.1 Creating cross table 
```
pd.crosstab(df['variable 1'],df['variable 2'])
```

#### 4.3.2 Using chi-square test
```
chi2_contingency(pd.crosstab(df['variable 1'],df['variable 2']))
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




