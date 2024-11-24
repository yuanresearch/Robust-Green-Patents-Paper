###############################
#### Set up the server:
#### apt-get update
#### apt install unzip
#### pip3 install pandas

###############################
#### mkdir PatentsViewRaw
#### mkdir tmp
#### mkdir Results
#### cd PatentsViewRaw

###############################
## Download key file from the patentsview website. https://patentsview.org/download/data-download-tables

# wget https://s3.amazonaws.com/data.patentsview.org/download/g_patent.tsv.zip && unzip g_patent.tsv.zip && rm g_patent.tsv.zip &&
# wget https://s3.amazonaws.com/data.patentsview.org/download/g_cpc_current.tsv.zip && unzip g_cpc_current.tsv.zip && rm g_cpc_current.tsv.zip



# Importing necessary packages.

import pandas as pd
import csv

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

Desktop_Developing = False
#Desktop_Developing = True ## If you are developing on your desktop, set it to True. Otherwise, set it to False.


### Read Patent.tsv. As the Oct/18/2024, most recent data is till June/30/2024.


patentRaw = "PatentsViewRaw/g_patent.tsv"

if Desktop_Developing:
    df_patentRaw = pd.read_csv(patentRaw, delimiter="\t", quoting = csv.QUOTE_NONNUMERIC, nrows=1000000)
else:
    df_patentRaw = pd.read_csv(patentRaw, delimiter="\t", quoting=csv.QUOTE_NONNUMERIC)

df_patentRaw = df_patentRaw[pd.to_numeric(df_patentRaw['patent_id'], errors='coerce').notnull()] ## Very important
df_patentRaw['patent_id'] = df_patentRaw['patent_id'].astype(int)
df_patentRaw['num_claims'] = df_patentRaw['num_claims'].astype(int)
df_patentRaw['withdrawn'] = df_patentRaw['withdrawn'].astype(int)

print (df_patentRaw.head(5)) ## Check the first 5 rows of the dataset.

#### Select valid patents. Here I keep utility patents.

df_patentRaw = df_patentRaw.loc[df_patentRaw['withdrawn'] < 1]
df_patentRaw = df_patentRaw.loc[df_patentRaw['patent_date'] >= '1976-01-01']
df_patentRaw = df_patentRaw.loc[df_patentRaw['patent_date'] <= '2024-07-01']
df_patentRaw = df_patentRaw.loc[df_patentRaw['patent_type'] == 'utility']

print (df_patentRaw.head(5)) ## Check the first 5 rows of the dataset.



#### I will drop abstract column to save memory.

# df_patentRaw = df_patentRaw.drop(['patent_abstract'], axis=1)


print ("Now I am getting all patents number!")
print (df_patentRaw)
print (len(df_patentRaw.index))

# df_patentRaw.to_csv('tmp/0_df_patentRaw.csv', index=False)




### Read CPC current dataset
cpc_current_Raw = "PatentsViewRaw/g_cpc_current.tsv"


if Desktop_Developing:
    df_cpc_current = pd.read_csv(cpc_current_Raw, delimiter="\t", quoting = csv.QUOTE_NONNUMERIC, nrows=2000000)
else:
    df_cpc_current = pd.read_csv(cpc_current_Raw, delimiter="\t", quoting = csv.QUOTE_NONNUMERIC)

# Convert the filtered 'patent_id' column to integers
df_cpc_current['patent_id'] = df_cpc_current['patent_id'].astype(int)
df_cpc_current['cpc_sequence'] = df_cpc_current['cpc_sequence'].astype(int)

print (df_cpc_current.head(5))
print (len(df_cpc_current.index))

print ('Hey, I am getting all CPCs number!')


### However, I only need the CPCs within selected time range. This also helps to decrease the memory usage.
col_one_list = df_patentRaw['patent_id'].tolist()
print (len(col_one_list))
df_cpc_current=df_cpc_current.loc[df_cpc_current['patent_id'].isin(col_one_list)]
print (df_cpc_current)
print (len(df_cpc_current.index))


### I will need to rearrange the sequence.
df_cpc_current_sorted = df_cpc_current.sort_values(['patent_id','cpc_sequence'],ascending=True)
print (df_cpc_current_sorted)

# df_cpc_current_sorted.to_csv('tmp/0_df_cpc_current_sorted.csv', index=False)

### get CPCs. The cpc_subclass contains the detailed information of CPCs. I will need
### to merge cpc_current_Raw with df_cpc_current_sorted to get the detailed information.

df_cpc_current_grouped_joined = df_cpc_current_sorted.groupby('patent_id', as_index=False).agg({'cpc_subclass' : '|'.join, 'cpc_sequence' : 'first'})
df_cpc_current_grouped_joined['all_CPC'] = df_cpc_current_grouped_joined['cpc_subclass']
df_cpc_current_grouped_joined = df_cpc_current_grouped_joined.join(df_cpc_current_grouped_joined['cpc_subclass'].str.split('|', expand=True).add_prefix('CPC'))
# df_cpc_current_grouped_joined.to_csv('tmp/0_df_cpc_current_grouped_joined.csv', index=False)
print (len(df_cpc_current_grouped_joined.index))

print ("Now start merging!")

#### Merge the selected patents and its CPC. Here is the foundation of our datasets.

df_patentRaw['patent_id'] = df_patentRaw['patent_id'].astype(int)
df_cpc_current_grouped_joined['patent_id'] = df_cpc_current_grouped_joined['patent_id'].astype(int)

df_merged_cpc = pd.merge(df_patentRaw ,df_cpc_current_grouped_joined,
                         left_on ='patent_id', right_on= 'patent_id', how = 'left')

print (len(df_patentRaw.index))
print (len(df_cpc_current_grouped_joined.index))
print (len(df_merged_cpc.index))

df_merged_cpc.to_csv('Results/0_RawPatents.csv', index=False)

### Now the data is ready for the next step.



