---
title: "Titanic Classification"
excerpt: "Classification Model For the Titanic Dataset 1<br/><img src='/images/titanic_1.jpg' style="width:500px;height:600px"> 
collection: portfolio
---


## Introduction
The titanic dataset seems to be the 'Hello World' of machine learning and is a collation of data about the individuals aboard the titanic and their survival.

The Titanic was passenger liner that sank in 1912 on the Atlantic ocean on voyage from Southhampton to New York City. Of the 2224 passengers and crew, over 67% passed away making it one of the worst maritime disasters ever.

### Problem Frame
The objective here is to predict the survival of a passenger when given their details only. The data set is already split by Kaggle and only data cleaning and model training are required.

### Performance Measure
Seeing as this is a classification problem, percentage accuracy will be a suitable measure of performance on the test set.

### Assumption Checks
Clarifying deliverables and data 

## Data Acquisition 
For the first step we just download the train and test data from [Kaggle](https://www.kaggle.com/c/titanic/data). 

### Create Workspace & Tools
Now we import the standard ML libraries for basic data wrangling.


```python
import numpy as np
import pandas as pd
print("Complete")
```

    Complete
    

### Load Data




```python
from google.colab import drive
drive.mount("/content/drive")
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
cd drive/My Drive/Developmental/Programming/Data Science/Projects/Kaggle: Titanic Classification

```

    [Errno 2] No such file or directory: 'drive/My Drive/Developmental/Programming/Data Science/Projects/Kaggle: Titanic Classification'
    /content/drive/My Drive/Developmental/Programming/Data Science/Projects/Kaggle: Titanic Classification
    


```python

train_set = pd.read_csv("train.csv")
test_set = pd.read_csv("test.csv")
print ("Data Loaded")
```

    Data Loaded
    


```python
test_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Data Manipulation




### Data Check



```python
train_set.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



From the describe we can already start to get some insight into the demographic of the passengers.

The 50th percentile for 'Passenger Class' is 3rd class, indicating that a majority of passengers were 3rd class passengers. 

A mean age of 30 and 75 percentile of 38 years old means the ages are skewed to the younger ages (>40) and fits with the life expencancy in the early 1900s (40 years). It can also be seen that most people did not have a sibling or parent (SibSp 75% = 1, ParCh 75% = 0 ).

With a mean of £32 and 75 percentile of £31, we can see that the top 25% paid the total of the bottom 75% therefore the wealth distrubution seems to skew toward a select few.

Now to check the nature of the dataset.


```python
train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Each row represents a passenger and their details in the 12 columns. "Sex", "Ticket", "Cabin" and 'Embarked' are all string type and transforming them into numerical representation or one-hot encoded would be more appropriate for training our model. The variety of inputs or each attribute will determine if one-hot encodes will be an efficient solution.


```python
train_set["Sex"].value_counts()
```




    male      577
    female    314
    Name: Sex, dtype: int64




```python
train_set["Embarked"].value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64



"Sex" and '"Embarked" only have 2 and 3 categories respectively and one-hot encoding can be a viable numerical representation of this attribute.


```python
train_set["Cabin"].value_counts()
```




    G6             4
    C23 C25 C27    4
    B96 B98        4
    D              3
    F33            3
    F2             3
    E101           3
    C22 C26        3
    D20            2
    D33            2
    B22            2
    C65            2
    E24            2
    E121           2
    B49            2
    B58 B60        2
    D35            2
    D36            2
    E67            2
    C92            2
    B20            2
    E8             2
    E44            2
    B35            2
    B51 B53 B55    2
    F4             2
    C124           2
    C83            2
    C68            2
    B5             2
                  ..
    C32            1
    B30            1
    C103           1
    B50            1
    E10            1
    F38            1
    B39            1
    E34            1
    A34            1
    E50            1
    D10 D12        1
    D45            1
    E31            1
    E49            1
    A7             1
    C128           1
    B69            1
    T              1
    D37            1
    D46            1
    E12            1
    C91            1
    A6             1
    B3             1
    A32            1
    E36            1
    A31            1
    B82 B84        1
    C99            1
    C85            1
    Name: Cabin, Length: 147, dtype: int64




```python
train_set["Ticket"].value_counts()
```




    1601                  7
    CA. 2343              7
    347082                7
    CA 2144               6
    347088                6
    3101295               6
    S.O.C. 14879          5
    382652                5
    19950                 4
    LINE                  4
    4133                  4
    2666                  4
    17421                 4
    347077                4
    113781                4
    W./C. 6608            4
    349909                4
    PC 17757              4
    113760                4
    347742                3
    230080                3
    239853                3
    PC 17582              3
    24160                 3
    PC 17755              3
    F.C.C. 13529          3
    PC 17572              3
    13502                 3
    C.A. 34651            3
    110152                3
                         ..
    PC 17595              1
    27267                 1
    347063                1
    65304                 1
    9234                  1
    2687                  1
    2628                  1
    C.A. 29178            1
    250653                1
    330958                1
    31418                 1
    SOTON/O.Q. 3101305    1
    29751                 1
    386525                1
    239854                1
    11752                 1
    113786                1
    349236                1
    349233                1
    C.A. 18723            1
    336439                1
    2695                  1
    28664                 1
    112052                1
    19972                 1
    36209                 1
    349247                1
    350060                1
    113807                1
    244358                1
    Name: Ticket, Length: 681, dtype: int64



As a categorical attribute, "Ticket" and "Cabin" contains too many inputs to be one-hot encoded individually as a dense matrix will slow runtimes when training. However, a **sparse matrix** may be an option to explore.

Another option may be to **parse the first letter of the "Cabin"** and encode that. The "Ticket" attribute, on the other hand, doesn't seem to have any pattern to it and this may even introduce some noise into the analysis. Therein the **"Ticket" will be dropped**. 


### Data Preparation
Now we will create some custom transformers to be used on both the training and test datasets but before we start manipulating the data, Ill make a copy of the train data.


```python
train_cleaning = train_set.copy(deep=True)
```

### Data Completion


```python
train_cleaning.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    

With 891 entries, this data set is relatively small for machine learning standards. "Age", "Cabin" and "Embarked" all have null columns and the effect of this missing data and how to mitigate errors will have to be examined. "Cabin" is missing <77% of its data and the efficiacy of this feature will have to be examed.

For now we will fill the "Age" Nulls will be filled with the average, "Embarked" will be filled with the mode and "Fare" will be filled with median.

We will also drop the "PassengerID", "Name" and "Ticket".


```python
train_cleaning["Age"].fillna(train_cleaning["Age"].mean(), inplace=True)

train_cleaning["Embarked"].fillna(train_cleaning["Embarked"].mode()[0], inplace=True)

train_cleaning["Fare"].fillna(train_cleaning["Fare"].median(), inplace=True)

train_cleaning.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)



print("Nulls filled and columns dropped")

```

    Nulls filled and columns dropped
    

### Data Correcting
Although there are some values that are much above the average, these are not outliers per se and may be essential to some underlying relation. For this reason corrcting for outliers will be skipped for the dataset.

### Feature Engineering
No apparent feature engineeing or attribute creation which will greatlyincrease accurary is immediately apparent and may be considered later.

### Feature Convertion 
Now we convert the categorical data into numerical.


```python
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

train_cleaning['Embarked_Code'] = label.fit_transform(train_cleaning['Embarked'])


train_clean = pd.get_dummies(train_cleaning)
train_clean.drop(["Embarked_Code"], 1)
train_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked_Code</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Data Exploration & Visualisation
### Data Visualisation


```python
%matplotlib inline
import matplotlib.pyplot as plt
train_clean.hist(bins=50, figsize=(20,15))

```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f2d6f979f28>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67b89630>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67bafc88>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67b60320>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67b05978>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67b2cfd0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67ade668>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67a83cf8>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67a83d30>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67a579b0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67a0b048>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f2d67a306a0>]],
          dtype=object)




![png](images/output_32_1.png)


These histograms seem to corroborate the deductions made from 'train_set.describe()'. We can also see that there are more instances of deaths than survivial in the 'Survived' attribute and this will affect the training process. We also see that many attributes are tail heavy and  will have to be manipulated to a more bell shaped distribution


### Data Exploration

looking for correlations
Experimenting with attribute combinations



```python
corr_matrix = train_clean.corr()
corr_matrix['Survived'].sort_values(ascending=False)
```




    Survived         1.000000
    Sex_female       0.543351
    Fare             0.257307
    Embarked_C       0.168240
    Parch            0.081629
    Embarked_Q       0.003650
    SibSp           -0.035322
    Age             -0.069809
    Embarked_S      -0.149683
    Embarked_Code   -0.167675
    Pclass          -0.338481
    Sex_male        -0.543351
    Name: Survived, dtype: float64



There seems to be a weak correlation between the fare of a ticket and the survival of a passenger as well as a moderate negative correlation with the class of the passenger. Since both attributes relate to the financial commitment to the voyage it indicates that there may be a correlation for their financial standing. 

Surprisingly there seems to be no correlation with age even though the evacuation prioritised women and children. This relationship could be non-linear or may indeed be uncorrelated. A correlation with gender may appear after the 'Sex' attribute transformed.

To better visualuse any non linear relationships we will plot the relations.


```python
from pandas.plotting import scatter_matrix

attributes = ['Survived', 'Fare', 'Parch', 'SibSp', 'Age', 'Pclass', 'Sex_female', 'Sex_male']
scatter_matrix(train_clean[attributes], figsize=(12,8))
```


```python
corr_matrix = train_clean.corr()
corr_matrix['Survived'].sort_values(ascending=False)
```




    Survived         1.000000
    Sex_female       0.543351
    Fare             0.257307
    Embarked_C       0.168240
    Parch            0.081629
    Embarked_Q       0.003650
    SibSp           -0.035322
    Age             -0.069809
    Embarked_S      -0.149683
    Embarked_Code   -0.167675
    Pclass          -0.338481
    Sex_male        -0.543351
    Name: Survived, dtype: float64



This increase the relationship to a weak correlation and should improve the overall reliability of this attribute.


```python
train_clean.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    Survived         891 non-null int64
    Pclass           891 non-null int64
    Age              891 non-null float64
    SibSp            891 non-null int64
    Parch            891 non-null int64
    Fare             891 non-null float64
    Embarked_Code    891 non-null int64
    Sex_female       891 non-null uint8
    Sex_male         891 non-null uint8
    Embarked_C       891 non-null uint8
    Embarked_Q       891 non-null uint8
    Embarked_S       891 non-null uint8
    dtypes: float64(2), int64(5), uint8(5)
    memory usage: 53.2 KB
    

## Model Selection
### Model Exploration
Now we will train and evaluate a selection of models on the training set. We use use ```cross_validate``` to score and shortlist an algorithm to optimise.




```python
titanic = np.array(train_clean.drop(["Survived"],1))
titanic_labels = np.array(train_clean['Survived'])


```


```python
from sklearn import ensemble, linear_model, gaussian_process, naive_bayes, tree, discriminant_analysis, svm, neighbors, model_selection

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    ]


cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = titanic

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    
    cv_results = model_selection.cross_validate(alg, titanic, titanic_labels, cv  = cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    alg.fit(titanic, titanic_labels)
    
    row_index+=1

    
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
```


```python
import seaborn as sns

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'b')

plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
```

    /usr/local/lib/python3.6/dist-packages/seaborn/categorical.py:1428: FutureWarning: remove_na is deprecated and is a private function. Do not use.
      stat_data = remove_na(group_data)
    




    Text(0, 0.5, 'Algorithm')




![png](images/output_44_2.png)


From the bar graph we can see that the Gradient Boosting Classifier performed best and would be the best suited model for predictions.
