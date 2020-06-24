#Detailed Analysis can be found in the notebook

import pandas as pd
import numpy as np
import openpyxl 
import matplotlib.pyplot as plt
from scipy import stats
import researchpy

#Load all the datasets provided into respective dataframes

suicide = pd.read_csv('data/410suiciderate.data.csv')
cancer = pd.read_csv('data/405iiunder75mortalityratefromcancerconsideredpreventable.data.csv')
liver = pd.read_csv('data/406iiunder75mortalityratefromliverdiseaseconsideredpreventable.data.csv')
resp = pd.read_csv('data/407iiunder75mortalityratefromrespiratorydiseaseconsideredpreventable.data.csv')
cardio = pd.read_csv('data/404iiunder75mortalityratefromcardiovasculardiseasesconsideredpreventable.data.csv')
file = 'data/laregionlookup2012_tcm77-368555.xls'
xl = pd.ExcelFile(file)
region = xl.parse('LA_region_2012')

#Data Cleaning

def data_prep(dataframe):
    #First merge the respective disease dataset with the region dataset to obtain region codes for all areas that are not set to areatype "Region"
    df = pd.merge(dataframe, region, how='left', left_on = 'Area Code', right_on = 'la_code')
    #After merge, the column region_code, which comes from region dataset is checked for null values and replaced with original values if it is null 
    df.region_code.fillna(df['Area Code'], inplace=True)
    #Another merge is performed to assign each region code to its respective region name
    df = pd.merge(dataframe, region, how='left', left_on = 'Area Code', right_on = 'region_code')
    #Only those rows are carried forward that have the following region_codes. Any blanks and nulls are eliminated
    region_codes = ['E12000001','E12000002','E12000003','E12000004','E12000005','E12000006','E12000007','E12000008','E12000009']
    df = df[df['region_code'].isin(region_codes)]
    #Only rows which have Sex = Male or Sex = Female are carried forward, eliminating rows with Sex = Persons
    df = df[df.Sex != 'Persons']
    #The time period column contains some dates in the mmm-yy format and some dates where it is Year1 - Year 5, eg 2012-14
    df['Year'], df['month'] = df['Time period'].str.split('-', 1).str
    df["month"] = pd.to_numeric(df["month"])
    indexNames = df[df['month'] > 12 ].index
    df.drop(indexNames , inplace=True)

    #Some of the dataframes had null values for the following 3 columns.
    #However, if rows containing null values for these columns are dropped, less than 50% data will remain for analysis, leading to incorrect conclusions
    #Hence, imputed null vaues in these columns with median. The reason for selecting median is that they are not affected by outliers.
    df['Value'] = df['Value'].fillna((df['Value'].median()))
    df['Lower CI limit'] = df['Lower CI limit'].fillna((df['Lower CI limit'].median()))
    df['Upper CI limit'] = df['Upper CI limit'].fillna((df['Upper CI limit'].median()))
    #Returning cleaned dataframe
    return df

#Exploratory Analysis

#Function defined for distribution of disease rate and their respective confidence limits across regions. 
def rates_overview_region(dataframe):
    df=dataframe.groupby(['region_name'])["Value", "Lower CI limit","Upper CI limit"].apply(lambda x : x.astype(int).sum())
    df[-15:].plot( y=['Value', 'Lower CI limit','Upper CI limit'], figsize=(20,5), grid=True)

#Distribution of Disease by Region
def dis_reg(dataframe):
    df_g1 = dataframe.groupby('region_name')['Value'].sum()
    df2 =df_g1.to_frame()
    df2.reset_index(inplace=True)
    df2.columns = ['region_name','Value']
    df2.plot(kind='bar',x='region_name',y='Value')
    plt.title('Distribution of Disease by Region')
    plt.show()

#Distribution of Disease by Gender
def dis_gender(dataframe):
    df_g2 = dataframe.groupby('Sex')['Value'].sum()
    df2 =df_g2.to_frame()
    df2.reset_index(inplace=True)
    df2.columns = ['Sex','Value']
    df2.plot(kind='bar',x='Sex',y='Value')
    plt.title('Distribution of Disease by Gender')
    plt.show()

#Function to visualize distribution of disease rates across gender and region 
def dis_gender_reg(datframe):
    datframe.groupby(['Sex','region_name'])['Value'].sum().unstack('Sex').plot.bar()
    plt.title('Distribution of Disease by Region and Gender')

#Progression of disease over time
def time_prog(dataframe):
    plt.rcParams["figure.figsize"] = (10,10)
    dataframe.groupby(['region_name','Year'])['Value'].sum().unstack('region_name').plot.line()
    plt.title('Progression of disease over years')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=1, borderaxespad=0.)


#Hypothesis Testing

#Function to create two datasets male and female to study the difference in mean
def diff_df(dataframe):
    male = dataframe[dataframe['Sex'] == 'Male'] 
    male.reset_index(inplace= True)
    male.rename(columns={'Value': 'ValueM'}, inplace=True)
    female = dataframe[dataframe['Sex'] == 'Female'] 
    female.reset_index(inplace= True)
    female.rename(columns={'Value': 'ValueF'}, inplace=True)
    diff = male['ValueM'] - female['ValueF']
    return diff

#Function to plot p-p plot to test if the data is normally distributed
def pplot(dataframe):
    stats.probplot(dataframe, plot= plt)
    plt.title('Gender P-P Plot') 
    plt.show()

#Function to further validate normal distribution through histograms
def hist(dataframe):
    dataframe.plot(kind= "hist", title= "Disease Values Histogram")
    plt.xlabel("Value")
    plt.show()

#Function to perform ttest
#Null hypothesis: No difference in suicide rates across Males and Females 
#Alternate hypothesis: Difference in suicide rates across Males and Females
def ttest(dataframe):
    male = dataframe[dataframe['Sex'] == 'Male'] 
    male.reset_index(inplace= True)
    male.rename(columns={'Value': 'ValueM'}, inplace=True)
    female = dataframe[dataframe['Sex'] == 'Female'] 
    female.reset_index(inplace= True)
    female.rename(columns={'Value': 'ValueF'}, inplace=True)
    descriptives, results = researchpy.ttest(male['ValueM'], female['ValueF'])
    print(descriptives)
    print(results)

#Data Cleaning
df_suicide=data_prep(suicide)
df_suicide.drop(['Category Type', 'Age', 'Count','Denominator','Area Code', 'Category','Value note','la_code','la_name','month'], axis=1, inplace=True)

df_cancer = data_prep(cancer)
df_cancer.drop(['Category Type', 'Age', 'Count','Denominator','Area Code', 'Category','Value note','la_code','la_name','month'], axis=1, inplace=True)

df_liver = data_prep(liver)
df_liver.drop(['Category Type', 'Age', 'Count','Denominator','Area Code', 'Category','Value note','la_code','la_name','month'], axis=1, inplace=True)

df_resp = data_prep(resp)
df_resp.drop(['Category Type', 'Age', 'Count','Denominator','Area Code', 'Category','Value note','la_code','la_name','month'], axis=1, inplace=True)

df_cardio = data_prep(cardio)
df_cardio.drop(['Category Type', 'Age', 'Count','Denominator','Area Code', 'Category','Value note','la_code','la_name','month'], axis=1, inplace=True)

#Data Visualization

rates_overview_region(df_suicide)
dis_reg(df_suicide)
dis_gender(df_suicide)
dis_gender_reg(df_suicide)
time_prog(df_suicide)
