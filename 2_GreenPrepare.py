#pip3 install -U scikit-learn
#pip3 install hdbscan

# Importing necessary packages.
import sys
import os
import zipfile as zip
import pandas as pd
import csv
import numpy as np

# wget https://s3.amazonaws.com/data.patentsview.org/download/g_patent_abstract.tsv.zip && unzip g_patent_abstract.tsv.zip && rm g_patent_abstract.tsv.zip


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)

Y_category ='Y02T'

USPTO_Green_address = 'Results/Step1/1_df_patent_USPTO_Greens_'+str(Y_category)+'.csv'

df_USPTO_Green = pd.read_csv(USPTO_Green_address, header=0, dtype='unicode', low_memory=False)

print (len(df_USPTO_Green.index))

##### Read CSV files:

df_patent_All_CPC = pd.read_csv('Results/0_RawPatents.csv', header=0, dtype='unicode', low_memory=False)

df_patent_All_CPC = df_patent_All_CPC[df_patent_All_CPC['all_CPC'].notna()]

## Now I need to get the top 10 most frequent CPCs in CPC0 category using value_counts() function.

print ('most frequent CPCs in CPC0 category in df_patent_All_CPC')

print (df_USPTO_Green['CPC0'].value_counts().head(100))

FirstCPC = df_USPTO_Green['CPC0'].value_counts().head(30).index.tolist()  ## NO LONGER USED!

print (FirstCPC)

### tell me the frequency of the CPC code: G08G in CPC0 category in df_USPTO_Green['CPC0'].value_counts()

print ('G08G-likewise expert defined CPC category counts')

print (df_USPTO_Green['CPC0'].value_counts()['C10L'])
print (df_USPTO_Green['CPC0'].value_counts()['B29D'])
print (df_USPTO_Green['CPC0'].value_counts()['F25B'])
print (df_USPTO_Green['CPC0'].value_counts()['G08G'])
print (df_USPTO_Green['CPC0'].value_counts()['H05B'])

# print (df_USPTO_Green['CPC0'].value_counts()['C01B'])
# print (df_USPTO_Green['CPC0'].value_counts()['B63J'])
# print (df_USPTO_Green['CPC0'].value_counts()['B60H'])

# Now I am creating a list of potentialGreen CPC category based on the expert defined CPC category.

potentialGreenListFromExpert = ['C10L', 'B29D', 'F25B', 'G08G', 'H05B']


## Now I will need to get the patents that contain these potentialGreen CPC category from df_patent_All_CPC. This is for later inference.

df_USPTO_Green_potentialGreen = df_patent_All_CPC[df_patent_All_CPC['CPC0'].str.contains('|'.join(potentialGreenListFromExpert), regex=True)]

print (len(df_USPTO_Green_potentialGreen.index))

##Now I am reading g_patent_abstract.tsv

df_patent_abstract = pd.read_csv('PatentsViewRaw/g_patent_abstract.tsv', delimiter="\t", quoting = csv.QUOTE_NONNUMERIC, low_memory=False)

## I need to get abstract for df_USPTO_Green and df_USPTO_Green_potentialGreen, respectively.

df_USPTO_Green = pd.merge(df_USPTO_Green, df_patent_abstract, how='left', left_on='patent_id', right_on='patent_id')

df_USPTO_Green_potentialGreen = pd.merge(df_USPTO_Green_potentialGreen, df_patent_abstract, how='left', left_on='patent_id', right_on='patent_id')

### I will only keep patent_id, patent_date, patent_title, patent_abstract, all_CPC, CPC0 in df_USPTO_Green and df_USPTO_Green_potentialGreen

df_USPTO_Green = df_USPTO_Green[['patent_id', 'patent_date', 'patent_title', 'patent_abstract', 'all_CPC', 'CPC0']]

df_USPTO_Green_potentialGreen = df_USPTO_Green_potentialGreen[['patent_id', 'patent_date', 'patent_title', 'patent_abstract', 'all_CPC', 'CPC0']]


###Key part. I need to remove the patents that already in df_USPTO_Green from df_USPTO_Green_potentialGreen

df_USPTO_Green_potentialGreen = df_USPTO_Green_potentialGreen[~df_USPTO_Green_potentialGreen['patent_id'].isin(df_USPTO_Green['patent_id'])]

print (len(df_USPTO_Green.index))

print (len(df_USPTO_Green_potentialGreen.index))

print (df_USPTO_Green.head(5))

print (df_USPTO_Green_potentialGreen.head(5))

## Now I am going to save these two files into Results/Step2 folder. with Y_category as the name.

df_USPTO_Green.to_csv('Results/Step2/'+str(Y_category)+'_USPTO_Green.csv', index=False)

df_USPTO_Green_potentialGreen.to_csv('Results/Step2/'+str(Y_category)+'_USPTO_Green_potentialGreen.csv', index=False)



