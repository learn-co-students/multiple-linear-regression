
# Module 1 - multiple regression

## Learning goals:

For a multivariable linear regression, students will be able to:

* compare and contrast with univariable linear regression
* write an example of the equation
* develop one with statsmodels 
* assess the model fit 
* validate the model


### Keyterms
- Multivariable
- Train-test split
- MSE: Mean squared error
- RSME: Root squared mean error


## Scenario
We are given a dataset with 8 features and must use them to predict the target the variable.

A real world example here is best. Some ideas:
- http://www.statsci.org/data/general/punting.html
- http://www.statsci.org/data/general/cherry.html
- https://www.kaggle.com/dongeorge/beer-consumption-sao-paulo

### Prior Knowledge

Linear regression with a single predictor variable. Here, you'll explore how to perform linear regressions using multiple independent variables to better predict a target variable.


### Discussion here
>

### Load Libraries and load in data


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split

import matplotlib.pyplot as plt
```


```python
df = pd.read_csv("./sample-data.csv")
```


```python
df.head()
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
      <th>feature_1</th>
      <th>feature_2</th>
      <th>feature_3</th>
      <th>feature_4</th>
      <th>feature_5</th>
      <th>feature_6</th>
      <th>feature_7</th>
      <th>feature_8</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.624542</td>
      <td>-0.288868</td>
      <td>-1.187570</td>
      <td>0.984566</td>
      <td>0.153860</td>
      <td>-0.766511</td>
      <td>-1.080118</td>
      <td>-0.832127</td>
      <td>-7.678545</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.450551</td>
      <td>0.089796</td>
      <td>-1.305782</td>
      <td>0.456512</td>
      <td>-0.121468</td>
      <td>0.351001</td>
      <td>-0.415352</td>
      <td>-1.532619</td>
      <td>-10.345242</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.366481</td>
      <td>-0.441462</td>
      <td>0.365679</td>
      <td>0.509546</td>
      <td>1.202953</td>
      <td>0.059840</td>
      <td>0.949623</td>
      <td>0.289978</td>
      <td>113.885412</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.354303</td>
      <td>-0.547035</td>
      <td>0.125080</td>
      <td>-0.326438</td>
      <td>-1.641818</td>
      <td>0.502364</td>
      <td>0.759431</td>
      <td>-1.761765</td>
      <td>-207.790271</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.217104</td>
      <td>0.960210</td>
      <td>-0.105948</td>
      <td>-1.385835</td>
      <td>1.338789</td>
      <td>-1.421409</td>
      <td>0.437848</td>
      <td>0.170899</td>
      <td>65.349023</td>
    </tr>
  </tbody>
</table>
</div>



### In this dataset what is the difference vs linear regression from before?
#### Discussion here

>

### Everyone write an example of an equation for our multilinear regression
**Send your equations to me via zoom or slack and I will paste them into the notebook**

Equations here

>

### Assessment:

In groups of 2 or 3 write a synopsis of the following summary

* What can you say about the coefficients?
* What do the p-values tell us?
* What does R^2 represent
* What other insights do you notice?


```python
formula = 'target~{}'.format("+".join(df.columns[:-1]))
formula
```




    'target~feature_1+feature_2+feature_3+feature_4+feature_5+feature_6+feature_7+feature_8'




```python
model = sm.OLS(df.target, df.drop('target', axis=1)).fit()
```


```python
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>target</td>      <th>  R-squared:         </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>2.661e+32</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 14 Apr 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>11:16:24</td>     <th>  Log-Likelihood:    </th>  <td>  28722.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  1000</td>      <th>  AIC:               </th> <td>-5.743e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   992</td>      <th>  BIC:               </th> <td>-5.739e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>feature_1</th> <td> 1.832e-14</td> <td> 2.67e-15</td> <td>    6.854</td> <td> 0.000</td> <td> 1.31e-14</td> <td> 2.36e-14</td>
</tr>
<tr>
  <th>feature_2</th> <td>   56.1836</td> <td>  2.6e-15</td> <td> 2.16e+16</td> <td> 0.000</td> <td>   56.184</td> <td>   56.184</td>
</tr>
<tr>
  <th>feature_3</th> <td>    4.8857</td> <td> 2.58e-15</td> <td> 1.89e+15</td> <td> 0.000</td> <td>    4.886</td> <td>    4.886</td>
</tr>
<tr>
  <th>feature_4</th> <td>   44.2428</td> <td> 2.63e-15</td> <td> 1.68e+16</td> <td> 0.000</td> <td>   44.243</td> <td>   44.243</td>
</tr>
<tr>
  <th>feature_5</th> <td>   88.5640</td> <td>  2.6e-15</td> <td> 3.41e+16</td> <td> 0.000</td> <td>   88.564</td> <td>   88.564</td>
</tr>
<tr>
  <th>feature_6</th> <td>   34.2871</td> <td> 2.56e-15</td> <td> 1.34e+16</td> <td> 0.000</td> <td>   34.287</td> <td>   34.287</td>
</tr>
<tr>
  <th>feature_7</th> <td> 5.862e-14</td> <td> 2.54e-15</td> <td>   23.049</td> <td> 0.000</td> <td> 5.36e-14</td> <td> 6.36e-14</td>
</tr>
<tr>
  <th>feature_8</th> <td>   19.8909</td> <td> 2.63e-15</td> <td> 7.57e+15</td> <td> 0.000</td> <td>   19.891</td> <td>   19.891</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td> 0.744</td> <th>  Durbin-Watson:     </th> <td>   1.969</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.689</td> <th>  Jarque-Bera (JB):  </th> <td>   0.815</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.059</td> <th>  Prob(JB):          </th> <td>   0.665</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 2.926</td> <th>  Cond. No.          </th> <td>    1.15</td>
</tr>
</table>



### Build LinReg Model with Scikit-Learn


```python
linreg = LinearRegression()
```


```python
X = df.drop("target", axis=1)
y = df.target
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
```


```python
linreg.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
             normalize=False)




```python
linreg.score(X_test, y_test)
```




    1.0



### Cross Validate model


```python
cv_linreg = cross_val_score(linreg, X, y, cv=5, n_jobs=-1)
cv_linreg
```




    array([1., 1., 1., 1., 1.])


