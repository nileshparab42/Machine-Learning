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

### Import Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
```

### Load Data Set
#### Loading xlsx files 
``` 
os.chdir("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df = pd.read_excel("Data Preprocessing Data File.xlsx")	
print(df)
```

#### Loading csv files 
``` 
os.chdir("C:\\Users\\Nilesh\\Documents\\GitHub\\Machine-Learning\\Data Set\\")
df = pd.read_csv("Data Preprocessing Data File.csv")
print(df)
```

### Identifying the dimensions

#### Shape of the dataset
```
df.shape
```

#### Columns in the dataset
```
df.columns
```

#### Top values of the dataset
```
df.head()
```

## 2 Variable identification

#### Identifying the datatypes of the variables
```
df.dtypes
```

#### Segmenting variables according to datatypes
```
categorical = []
contineous = []
for column in df.columns:
    if df[column].dtypes == "int64":
        contineous.append(column)
    if df[column].dtypes == "object":
        categorical.append(column)
```

## 3 Plotting multiple graphs
```
plt.rcParams["figure.figsize"] = [13.50, 3.50]
plt.rcParams["figure.autolayout"] = True
figure, axis = plt.subplots(1, 3)
  
df['variable1'].plot.hist(title="Title1",ax=axis[0],color="MEDIUMVIOLETRED")
df['variable2'].plot.hist(title="Title2",ax=axis[1],color="INDIGO")
df['variable3'].plot.hist(title="Title3",color="MEDIUMVIOLETRED")
```


## 4 Univarient Analysis

### Univarient Analysis for Contineous Variables

#### Tabular Methods
```
df.describe()
```

#### Graphical method
##### Plotting histogram
```
df['variable'].plot.hist()
```

##### Plotting boxplot
```
df['variable'].plot.box()
```

## Univarient Analysis for Categorical Variable

### Tabular Methods
#### Creating frequency table
```
df['variable'].value_counts() 
```

#### Create percentages frequencies 
```
df['variable'].value_counts()/len(df['variable']) 
```
### Graphical method
#### Plotting barplot 
```
df['variable'].value_counts().plot.bar() 
```

#### Plotting barplot using percentage 
```
(df['variable'].value_counts()/len(df['variable'])).plot.bar() 
```


## 5 Bivarient Analysis

### Contineous - Contineous Variable

#### Plotting scatter plot 
```
df.plot.scatter('variable_1','variable_2')
```

#### Using correlation matrix
```
df.corr()
``` 
#### Plotting heatmap 
```
dataplot = sns.heatmap(df.corr(), cmap=sns.cubehelix_palette(as_cmap=True), annot=True)
```
### Categorical - Contineous Variable
#### Using tabular method
```
df.groupby('Categorical_variable')['contineous_variable'].mean()
```

#### Creating bar plot
```
df.groupby('Categorical_variable')['contineous_variable'].mean().plot.bar
```

#### Creating box plot
```
df.boxplot('contineous_variable',by='categorical_variable')
```

#### By using t-test  
Create groups of categorical variables
```
group = df[df['categorical_variable']=='Value']
```
Calculating t-test for each categorical variable
```
ttest_ind[group1['contineous_variable'],,nan-policy='omit']
```
#### By using ANOVA
```
new = ols('contineous_variable ~ categorical_variable',data=df).fit()
an = sm.stats.anova_lm(new,typ=2)
an
```

### Categorical - Categorical Variable
#### Creating cross table 
```
pd.crosstab(df['variable 1'],df['variable 2'])
```

#### Using chi-square test
```
chi2_contingency(pd.crosstab(df['variable 1'],df['variable 2']))
```

## 6 Missing Value Treatment

### Identifying Missing Values
#### Using describe function
```
df.describe()
```
#### Using isnull function
```
df.isnull().isnull()
```

### Treatment of missing values
#### Dropping rows whenever there are missing values
```
df.dropna().shape
```
#### Dropping rows where all the entries are missing
```
df.dropna(how='all').shape
```
#### Dropping columns whenever there are missing values
```
df.dropna(axis=1).shape
```
#### Dropping columns where all the entries are missing
```
df.dropna(axis=1,how='all').shape
```
#### Filling all the missing values with constant
```
df.fillna(const)
```
#### Filling all the missing values with mean
```
df['Variable'].fillna(df['variable'].mean)
```
#### Filling all the missing values median
```
df['Variable'].fillna(df['variable'].median)
```
#### Filling all the missing values with mode
```
df['Variable'].fillna(df['variable'].mode)
```

## 7 Outliers Treatement

### Univarient Outlier Detection
#### Creating boxplot 
```
df['Variable'].plot.box()
```
#### Using IQR Method
```
Q1 = df.Variable.quantile(0.25)
Q2 = df.Variable.quantile(0.75)
IQR = Q3-Q1
lower_limit = Q1-1.5*IQR
upper_limit = Q3+1.5*IQR
```

### Bivarient Outlier Detection
#### Creating scatter plot 
```
df['Variable'].plot.box()
df.plot.scatter('Variable_1','Variable_2')
```

### Removing Outliers

#### Removing values below or above range
```
df = df[df['Variable']<inside>range]
```
#### Replacing outliers with mean values
```
df.loc[df['Variable']<outside>range,'Variable'] = np.mean(df['Variables'])
```

##  8 Export Preprocessed Dataset
```
df.to_csv('Preprocessed.csv', encoding='utf-8', index=False)
```




