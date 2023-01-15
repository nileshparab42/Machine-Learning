![Cover image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/DP-Cover.png)

# Data Preprocessing

Data preprocessing is the process of cleaning, transforming, and organizing the raw data before feeding it into a machine learning model. It is an important step in the machine learning pipeline as it can greatly affect the performance of the model. Data preprocessing includes tasks such as handling missing values, removing outliers, converting categorical variables to numerical, normalizing the data, and splitting the data into training and testing sets. The goal of data preprocessing is to make the data suitable for the machine learning model by making it more informative and reducing the noise in the data. Additionally it also helps to avoid errors in data and overfitting.

## Reading the data
![Variable image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Variable.png)

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

Pandas is a powerful library in Python that is commonly used for data manipulation and analysis. One of the most basic tasks in Pandas is reading data into a DataFrame.

To read data into a DataFrame, you can use the `pd.read_csv()` function, which reads a CSV file into a DataFrame. The first argument of the function is the file path, and additional parameters can be used to specify options such as the separator, header, and index column.

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
Once the data is read into a DataFrame, you can use various Pandas functions to manipulate and analyze the data.

### Identifying the dimensions

In Pandas, the dimensions of a DataFrame or Series can be identified by using the `shape` attribute.

The shape attribute returns a tuple of (number of rows, number of columns) for a DataFrame, and (number of rows) for a Series.

#### Shape of the dataset
```
df.shape
```

You can also use `len()` function to get number of rows in the dataframe
```
print(len(df)) 
```

#### Columns in the dataset
You can also use `df.columns` to get all the column names in the dataframe and `df.index` to get the index of the dataframe.
```
df.columns
df.index
```

#### Top values of the dataset

The `head()` function in Pandas is used to return the first n rows of a DataFrame or Series. The default number of rows returned is 5, but you can specify a different number by passing an integer as an argument.
```
df.head()
```
Knowing the dimensions of a DataFrame or Series is important for understanding the size and structure of the data, and for selecting the appropriate methods for manipulating and analyzing the data.


## Variable identification
![Variable image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Variable.png)

In machine learning, variable identification refers to the process of identifying and selecting the most relevant variables (also called features) from a dataset to be used in building a predictive model. The goal of variable identification is to select the variables that are most informative and have the strongest relationship with the target variable.

#### Identifying the datatypes of the variables

In Pandas, the `dtypes` attribute is used to check the data types of the columns in a DataFrame. The `dtypes` attribute returns a Series with the data types of each column in the DataFrame.

```
df.dtypes
```
It's important to check the data types of the columns in a DataFrame before performing any operations on them, as certain operations may only be applicable to certain data types. For example, you can't perform mathematical operations on string data.

You can also use `pd.DataFrame.astype()` method to change the data type of a column or multiple columns.
```
df['Variable'] = df['Variable'].astype(float)
```

#### Segmenting variables according to datatypes

*Continuous variables*, on the other hand, can take on any value within a certain range. These variables are numeric and can be divided into smaller units. For example, the variable "weight" can take on any numeric value, such as 150 pounds or 175 pounds. Continuous variables can be further divided into interval and ratio variables. Interval variables have an arbitrary zero point, such as temperature in Celsius, while ratio variables have a true zero point, such as weight in pounds. It's important to note that the distinction between categorical and continuous variables can be dependent on the problem and the context.

*Categorical variables* are variables that can be divided into separate categories. These variables can take on a limited number of values, and the values represent different categories. For example, the variable "color" can take on the values "red", "green", "blue", etc. and each value represents a different category of color. Categorical variables can be further divided into ordinal and nominal variables. Ordinal variables have a natural order or ranking, such as small, medium, and large, while nominal variables don't have any order.


```
categorical = []
continuous = []
for column in df.columns:
    if df[column].dtypes == "int64":
        continuous.append(column)
    if df[column].dtypes == "object":
        categorical.append(column)
```

It's important to note that variable identification is an iterative process and the variables selected may change as the model is developed, tested and refined.

## Plotting multiple graphs
![Plotting graph image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Plotting-Graph.png)

Matplotlib is a powerful library in Python for creating various types of plots and visualizations. When working with multiple plots, it's important to use subplots to organize the plots in a grid.

The `plt.subplots()` function can be used to create a figure and a set of subplots. The function takes two arguments: the number of rows and the number of columns of the subplot grid.
```
plt.rcParams["figure.figsize"] = [13.50, 3.50]
plt.rcParams["figure.autolayout"] = True
figure, axis = plt.subplots(1, 3)
```
This will create a 2x2 grid of subplots.

Once you have created the subplots, you can plot on each subplot using the `ax` object, which is an array of axes objects. You can access each subplot by indexing the `ax` object.
```
df['variable1'].plot.hist(title="Title1",ax=axis[0],color="MEDIUMVIOLETRED")
df['variable2'].plot.hist(title="Title2",ax=axis[1],color="INDIGO")
df['variable3'].plot.hist(title="Title3",color="MEDIUMVIOLETRED")
```
Finally, you can display the plots using `plt.show()`

You can also customize the appearance and layout of the subplots using various functions and properties such as title, x and y labels, legend, etc.

## 4 Univarient Analysis
![Univarient image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Univarient.png)

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

### Univarient Analysis for Categorical Variable

#### Tabular Methods
##### Creating frequency table
```
df['variable'].value_counts() 
```

##### Create percentages frequencies 
```
df['variable'].value_counts()/len(df['variable']) 
```
#### Graphical method
##### Plotting barplot 
```
df['variable'].value_counts().plot.bar() 
```

##### Plotting barplot using percentage 
```
(df['variable'].value_counts()/len(df['variable'])).plot.bar() 
```


## 5 Bivarient Analysis
![Bivarient image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Multivalue.png)

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
### Categorical - continuous Variable
#### Using tabular method
```
df.groupby('Categorical_variable')['continuous_variable'].mean()
```

#### Creating bar plot
```
df.groupby('Categorical_variable')['continuous_variable'].mean().plot.bar
```

#### Creating box plot
```
df.boxplot('continuous_variable',by='categorical_variable')
```

#### By using t-test  
Create groups of categorical variables
```
group = df[df['categorical_variable']=='Value']
```
Calculating t-test for each categorical variable
```
ttest_ind[group1['continuous_variable'],,nan-policy='omit']
```
#### By using ANOVA
```
new = ols('continuous_variable ~ categorical_variable',data=df).fit()
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
![Missing value image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Missing-Value.png)

Missing values in a dataset can greatly affect the performance of a machine learning model. Therefore, it's important to handle missing values appropriately before building the model. 

### Identifying Missing Values
#### Using describe function

The `describe()` function in Pandas is used to generate descriptive statistics of the numerical columns in a DataFrame. It returns a new DataFrame with the statistics, which includes count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum.
```
df.describe()
```
#### Using isnull function

The `isnull()` function in Pandas is used to check for missing values in a DataFrame or Series. It returns a new DataFrame or Series with the same shape as the original, but containing `True` for cells with missing values and `False` for cells with non-missing values.

You can also use `sum()` function on the output of `isnull()` function which will return the total number of missing values in each column
```
df.isnull().sum()
```

### Treatment of missing values

The `dropna()` function in Pandas is used to remove missing values from a DataFrame or Series. It returns a new DataFrame or Series with the missing values removed.

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
The `fillna()` function in Pandas is used to replace missing values in a DataFrame or Series with a specific value or using a method such as forward fill or backward fill. It returns a new DataFrame or Series with the missing values replaced.

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
The best approach to treating missing values depends on the specific problem and dataset. It's important to evaluate the impact of different methods on the performance of the model and choose the method that results in the best performance.

## 7 Outliers Treatement
![Outlier image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/Outlier.png)

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
In Pandas, you can use the `to_csv()` function to export a DataFrame to a CSV file. The `to_csv()` function takes several optional parameters such as the file name, delimiter, index, header, etc.
Here's an example of using the `to_csv()` function to export a DataFrame to a CSV file:
```
df.to_csv('Preprocessed.csv', encoding='utf-8', index=False)
```
You can also use the `to_excel()` function to export a DataFrame to an Excel file.
```
df.to_excel('Preprocessed.xlsx', index=False)
```
This will export the DataFrame to an excel file named "data.xlsx" in the current working directory.

Pandas also support exporting DataFrame to many other file formats like feather, parquet, sql, etc. You can use the appropriate function for the desired file format to export the data.



