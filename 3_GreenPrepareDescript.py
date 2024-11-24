
# os.environ["http_proxy"] = "http://localhost:8890"
# os.environ["https_proxy"] = "http://localhost:8890"
#


#### Importing necessary packages.
import sys
import os
import zipfile as zip
import pandas as pd
import csv
import numpy as np

#import openai

# # run it from command line to downnload detail description text files.
# for year in {1976..2024}; do
#     wget https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_${year}.tsv.zip &&
#     unzip g_detail_desc_text_${year}.tsv.zip &&
#     rm g_detail_desc_text_${year}.tsv.zip
# done

## For the training data and candidate dataset, we need descriptions from the raw files.


import pandas as pd
import os


Y_category ='Y02T'

# Read df_USPTO_Green and df_USPTO_Green_potentialGreen

df_USPTO_Green_address = 'Results/Step2/'+str(Y_category)+'_USPTO_Green.csv'

df_USPTO_Green = pd.read_csv(df_USPTO_Green_address, header=0, dtype='unicode', low_memory=False)

df_USPTO_Green_potentialGreen_address = 'Results/Step2/'+str(Y_category)+'_USPTO_Green_potentialGreen.csv'

df_USPTO_Green_potentialGreen = pd.read_csv(df_USPTO_Green_potentialGreen_address, header=0, dtype='unicode', low_memory=False)

## Create a grant_year column in df_USPTO_Green and df_USPTO_Green_potentialGreen

df_USPTO_Green['grant_year'] = pd.to_datetime(df_USPTO_Green['patent_date']).dt.year

df_USPTO_Green_potentialGreen['grant_year'] = pd.to_datetime(df_USPTO_Green_potentialGreen['patent_date']).dt.year

# The purpose of the next step, is to extract detailed descriptions from the raw files, based on patents ID and patent granted year from df_USPTO_Green and df_USPTO_Green_potentialGreen.

####

print("start to extract detailed descriptions from the raw files for df_USPTO_Green")

# Prepare a list to store matching descriptions
matching_descriptions_df_USPTO_Green = []

# Loop through each year in your data range
for year in range(1976, 2024 + 1):

    print (year)
#for year in range(1976, 1980):
    # Check if the year has patents in your list
    year_patents = df_USPTO_Green[df_USPTO_Green['grant_year'] == year]

    if year_patents.empty:
        continue  # Skip if no patents for this year

    # Construct the filename for the current year's data. filename stored at /home/yuansun/Desktop/GreenPrepareDescript

    filename = f'/home/yuansun/Desktop/GreenPrepareDescript/g_detail_desc_text_{year}.tsv'

    if os.path.exists(filename):
        # Load the TSV file for the current year
        year_data = pd.read_csv(filename, sep='\t', dtype=str)

        # Filter for matching patent numbers
        matching_data = year_data[year_data['patent_id'].isin(year_patents['patent_id'])]

        # Append to the list if there are matches
        if not matching_data.empty:
            matching_descriptions_df_USPTO_Green.append(matching_data)
    else:
        print(f"File {filename} does not exist.")

# Combine all matching descriptions into one DataFrame
if matching_descriptions_df_USPTO_Green:
    all_matches_df_USPTO_Green = pd.concat(matching_descriptions_df_USPTO_Green, ignore_index=True)

    ## Merge df_USPTO_Green with all_matches_df_USPTO_Green by patent_id

    all_matches_df_USPTO_Green = pd.merge(df_USPTO_Green, all_matches_df_USPTO_Green, how='left', left_on='patent_id', right_on='patent_id')

    # Save to a new CSV or process further

    all_matches_df_USPTO_Green.to_csv('Results/Step3/' + str(Y_category) + '_USPTO_Green.csv', index=False)
    print("Matching descriptions saved!")
else:
    print("No matching descriptions found.")


### Now do it for df_USPTO_Green_potentialGreen

print("start to extract detailed descriptions from the raw files for df_USPTO_Green_potentialGreen")

# Prepare a list to store matching descriptions

matching_descriptions_df_USPTO_Green_potentialGreen = []

# Loop through each year in your data range

for year in range(1976, 2024 + 1):

    print (year)
    # Check if the year has patents in your list

    year_patents = df_USPTO_Green_potentialGreen[df_USPTO_Green_potentialGreen['grant_year'] == year]

    if year_patents.empty:
        continue  # Skip if no patents for this year

    # Construct the filename for the current year's data. filename stored at /home/yuansun/Desktop/GreenPrepareDescript

    filename = f'/home/yuansun/Desktop/GreenPrepareDescript/g_detail_desc_text_{year}.tsv'

    if os.path.exists(filename):
        # Load the TSV file for the current year

        year_data = pd.read_csv(filename, sep='\t', dtype=str)

        # Filter for matching patent numbers

        matching_data = year_data[year_data['patent_id'].isin(year_patents['patent_id'])]

        # Append to the list if there are matches

        if not matching_data.empty:
            matching_descriptions_df_USPTO_Green_potentialGreen.append(matching_data)
    else:
        print(f"File {filename} does not exist.")

# Combine all matching descriptions into one DataFrame

if matching_descriptions_df_USPTO_Green_potentialGreen:
    all_matches_df_USPTO_Green_potentialGreen = pd.concat(matching_descriptions_df_USPTO_Green_potentialGreen, ignore_index=True)

    ## Merge df_USPTO_Green_potentialGreen with all_matches_df_USPTO_Green_potentialGreen by patent_id

    all_matches_df_USPTO_Green_potentialGreen = pd.merge(df_USPTO_Green_potentialGreen, all_matches_df_USPTO_Green_potentialGreen, how='left', left_on='patent_id', right_on='patent_id')

    # Save to a new CSV or process further

    all_matches_df_USPTO_Green_potentialGreen.to_csv('Results/Step3/' + str(Y_category) + '_USPTO_Green_potentialGreen.csv', index=False)
    print("Matching descriptions saved!")








