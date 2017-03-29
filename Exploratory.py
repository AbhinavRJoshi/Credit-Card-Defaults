import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading our data set from the excel file and setting the header to the first row of the input file
df = pd.read_excel("default of credit card clients.xls", header = 1 , index_col = 0 )

#Data Exploration

#Looking at the shape of our dataframe  we can see that there are 30,000 rows and 25 columns
df.shape

#From the info output we can see that we don't have any non-null values in our data frame
df.info()
df.describe()
#Taking a closer look at our dependent column default payment next month we can see that although it is a boolean value it is still stored as an integer
df['default payment next month'].head()

#Converting it into a boolean value
df['default payment next month'] = df['default payment next month'].astype('bool')

#Now that we have dependent variable as a categorical variable we will now look at the independent variables to delete, modify or combine
df['default payment next month'].hist(bins = 2)
plt.show()

#Now from this histogram we can see that the majority of people in our list are non-defaulters, we can also get an exact table
df['default payment next month'].value_counts()

#From the table we can see that we havev 23364 non-defaulters and 6636 defaulters

#Just looking at the % of defaulters in each education group we can see that people with just high school education have the highest proportion of defaulters
df.EDUCATION.value_counts()
df.groupby('EDUCATION')['default payment next month'].mean().plot()

#Now because we don't know what 0,4,5,6 in education stand for and they represent only a small portion of our dataset (468 rows) we are going to remove this rows
df = df.drop(df[df['EDUCATION'].isin([0,4,5,6])].index)


#Taking a look at the distribution of out age parameter we can see the average mean is 35.47 and the average median of the dataset is 34 and it is a fairly normal distribution with the center at 30
df.AGE.mean()
df.AGE.median()
df.AGE.hist()


#Now for the purpose of our machine learning model I want to create an additional column that is true when the persons age is greater than 35 and false otherwise
df['Greater than 35'] = False
df.loc[df.AGE>=35,'Greater than 35'] = True

#Since we now have a column that calculates whether the age is greater than 35 or not, we will remove our AGE column
df.drop('AGE',axis = 1 , inplace = True)
      

#Now our PAY_0 - PAY_6 columns give us the default status for each user for the past 6 months
#Rather than looking at all 6 columns I think it would be interesting if we simply used a an average value and the actual value of the last month
df['Pay Status Average'] = df[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].mean(axis = 1)
df.drop(['PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'],axis = 1 , inplace = True)      

#Similarly for the pay amount and bill amount columns I want to have a final sum column for each
df['Bill Total'] = df[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].sum(axis = 1)
df['Pay Total'] = df[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis = 1)

#Now that we have the total bill amounts and total pay amounts, let us create a new column that looks at the total default
df['Default_Total'] = df['Bill Total'] - df['Pay Total']


#Now we can remove our bill amounts and pay amounts columns as well
df.drop(['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'],axis = 1 , inplace = True)      
df.drop(['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6'],axis = 1 , inplace = True)      

#Creating a dictionary to map the integer values 
map_dict_gender = {1:"Male",2:"Female"}
df.SEX = df.SEX.map(map_dict_gender)

#Looking at the mean default payments by sex we can see that the proportion of men who have defaulted is slightly higher than that of a woman
df.groupby('SEX')['default payment next month'].mean()

#Taking a look at the mean values of both the dependent default and average pay status we see the same trend where men are more likely to default and be late on their payments than women
df.pivot_table(index = "SEX", values = ['default payment next month','Pay Status Average'],aggfunc =np.mean)
