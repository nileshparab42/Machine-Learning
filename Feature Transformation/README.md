![Cover image](https://github.com/nileshparab42/Machine-Learning/blob/master/assets/DP-Cover.png)

# Feature Transformation

Feature transformation refers to the process of modifying or transforming the input features of a dataset in order to improve the performance of a machine learning model. There are many different types of feature transformations, such as scaling, normalization, and dimensionality reduction. These techniques can help to improve the accuracy and efficiency of the model by reducing the impact of noise in the data and making it easier for the model to find patterns in the data.

## Encoding of Categorical variable

To find out how many unique labels are there in variable
```
len(df['Variable'].unique())
```

### Ordinal Variable

#### Label Encoding
```
le = LabelEncoder()
df['Variable'] = le.fit_transform(df['Variable'])

```

#### Target guided Encoding
```
ordinal_labels = df.groupby(['Variable'])['Target_variable'].mean().sort_values()index
ordinal_labels_dict = {k:i for i,k in enumerate(ordinal_labels,0)}
df['Encoded_Variable'] = df['Variable'].map(ordinal_labels_dict)
```


### Nominal Variable

#### One hot Encoding
```
pd.get_dummies(df).shape
```

#### Count or frequency encoding

To obtain the counts for each one of the labels in variable
let's capture this in a dictionary that we can use to re-map the labels
```
df_frequency_map = df.Variable.value_counts().to_dict()
```
now we replace Variable labels in the dataset
```
df.Variable = df.Variable.map(df_frequency_map)
```

#### One hot Encoding with many categories

Let's find the top 10 most frequent categories for the variable 
```
df.Variable.value_counts().sort_values(ascending=False)
```
Now we are making a list with the most frequent categoriesof the variable
```
top_10 = [x for x in df.Variable.value_counts().sort_values(ascending=False).head(10).index]
```
Then we make the 10 binary variables
```
for label in top_10:
    df[label] = np.where(df['Variable']==label,1,0) 
```
get whole set of dummy variables, for all the categorical variable
```
def one_hot_top_x(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable+_+label] = np.where(df[variable]==label,1,0) 
```
Using the function
```
one_hot_top_x(df,'Variable',top_10)
```

#### Mean Encoding
```
te = TargetEncoder()
te.fit(x=df['Variable'],y=df['Target_Variable'])
values = te.transform(df['Variable'])
df = pd.concat([df,values],axis=1)
```

## Feature transformation 

### Standardization 
```
scaler = StandardScaler()
Variable_scaled = scaler.fit_transform(Variable)
```

### Min-Max Scaling / Normalisation
```
min_max = MinMaxScaler()
df_minmax = pd.DataFrame(min_max.fit_transform(df),columns=df.columns)
```
### Robust Scaler
```
scaler = RobustScaler()
df_robust_scaled = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
```
### Guassian Transformation

If you want to check whether feature is guassian or normal distributed by using Q-Q plot
```
def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    df[feature].hist(color="MEDIUMVIOLETRED")
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist=normal,plot=pylab)
    plt.show
```
Ploting data for perticular variable
```
plot_data(df,'Variable')
```
Logarithmic Transformation
```
df['Variable_log'] = np.log(df['Variable'])
```
Reciprocal Transformation
```
df['Variable_reciprocal'] = 1/df.Variable
```
Square Root Transformation
```
df['Variable_sqrt'] = df.Variable**(1/2)
```
Exponential Transformation
```
df['Variable_expo'] = df.Variable**(1/1.2)
```
BoxCOx Transformation
```
df['Variable_Boxcox'],parameters = stats.boxcox(df['Variable'])
```
## Separating dependent and independent variables

Creating Independent Variable
```
X = df1.iloc[:,:-1].values
```

Create Dependent Variable
```
Y = df1.iloc[:,-1].values
```

## Test-train split
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, shuffle=True)
```





