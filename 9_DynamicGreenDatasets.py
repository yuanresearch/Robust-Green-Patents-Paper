
#### Importing necessary packages.
import sys
import os
import zipfile as zip
from time import sleep
import pandas as pd
import csv
import numpy as np
import pandas as pd
import os
import openai
from datasets import Dataset
import pandas as pd
import re

Y_category ='Y02T'
os.environ["http_proxy"] = "http://localhost:8890"
os.environ["https_proxy"] = "http://localhost:8890"

df_USPTO_Green_USPTO_Screened_address = 'Results/Step4/' + str(Y_category) + '_USPTO_Green.csv'

df_USPTO_Green_USPTO_Screened = pd.read_csv(df_USPTO_Green_USPTO_Screened_address, header=0, dtype='unicode', low_memory=False)

## save it.
df_USPTO_Green_USPTO_Original = df_USPTO_Green_USPTO_Screened[['patent_id', 'patent_date', 'patent_title', 'patent_abstract', 'all_CPC']].copy()

df_USPTO_Green_USPTO_Original.to_csv('Results/Step9/' + str(Y_category) + '_USPTO_Green_USPTO_Original.csv', index=False)


df_USPTO_Green_fine_tuned_address = 'Results/Step7/' + str(Y_category) + '_USPTO_Green_infer.csv'

df_USPTO_Green_fine_tuned = pd.read_csv(df_USPTO_Green_fine_tuned_address, header=0, dtype='unicode', low_memory=False)


### The Dynamic Green Dataset will consist of two parts, one is from the USPTO_Screened, and the other is from the fine-tuned LLM model.

### print the rows counts of df_USPTO_Green_USPTO_Screened, with its name

print ('USPTO Green patents counts')

print (len(df_USPTO_Green_USPTO_Screened.index))

df_USPTO_Green_USPTO_Screened['Decision'] = df_USPTO_Green_USPTO_Screened['gpt4_is_green_patent'].str.extract(r'Decision:\s*(\w+)')
df_USPTO_Green_USPTO_Screened['Confidence Score'] = df_USPTO_Green_USPTO_Screened['gpt4_is_green_patent'].str.extract(r'Confidence Score:\s*(\d+)')

df_USPTO_Green_USPTO_Screened['Confidence Score'] = df_USPTO_Green_USPTO_Screened['Confidence Score'].astype(float)


### when the Decision is Yes, and the confidence score is lareger or equal to 85, I believe it is a non-green patent. Otherwise, it is a green patent.
### Here I only need to keep the green patents.

df_USPTO_Green_USPTO_Screened = df_USPTO_Green_USPTO_Screened[(df_USPTO_Green_USPTO_Screened['Decision'] == 'Yes') & (
                df_USPTO_Green_USPTO_Screened['Confidence Score'] >= 85)].copy()

# df_USPTO_Green_USPTO_Screened = df_USPTO_Green_USPTO_Screened[(df_USPTO_Green_USPTO_Screened['Decision'] == 'Yes')].copy()

print ('USPTO Green patents counts screened by confidence score and is it recognized as green patent')

print (len(df_USPTO_Green_USPTO_Screened.index))

### I will only keep patent_id, patent_date, patent_title, patent_abstract, all_cpc in the dataset.

df_USPTO_Green_USPTO_Screened = df_USPTO_Green_USPTO_Screened[['patent_id', 'patent_date', 'patent_title', 'patent_abstract', 'all_CPC']]

## save it.

df_USPTO_Green_USPTO_Screened.to_csv('Results/Step9/' + str(Y_category) + '_USPTO_Green_USPTO_Screened.csv', index=False)


### Now I will process the fine-tuned LLM model data.


# List to store processed results
processed_results = []

for index, row in df_USPTO_Green_fine_tuned.iterrows():
    patent_id = row['patent_id']
    assistant_answer = row['assistant_answer']

    # Convert to string, or default to an empty string if NaN
    assistant_answer = str(assistant_answer) if pd.notna(assistant_answer) else ""


    # Extract Decision and Confidence Score
    # Look for "Yes" or "No" anywhere in the text (first occurrence)
    decision_match = re.search(r'(Yes|No)', assistant_answer, re.IGNORECASE)
    # Look for the first numerical value (assumed to be confidence score)
    confidence_match = re.search(r'\b(\d{1,3})\b', assistant_answer)

    decision = decision_match.group(0) if decision_match else ""
    confidence_score = confidence_match.group(0) if confidence_match else ""

    # Append processed data to the list
    processed_results.append({
        'patent_id': patent_id,
        'ft_llm_decision': decision,
        'ft_llm_confidence_score': confidence_score,
        'ft_llm_raw': assistant_answer
    })

# Convert to DataFrame and save as a new CSV
processed_df = pd.DataFrame(processed_results)
#processed_df.to_csv('Results/Step8/' + str(Y_category) + '_processed_df.csv', index=False)
print("df_USPTO_Green_fine_tuned Results Cleaned")

print ('df_USPTO_Green_fine_tuned patent counts')

print (len(df_USPTO_Green_fine_tuned.index))

### I will only keep infered patents when ft_llm_decision is Yes

df_USPTO_Green_fine_tuned = processed_df[(processed_df['ft_llm_decision'] == 'Yes')].copy()

print ('df_USPTO_Green_fine_tuned patent counts screened by decision')

print (len(df_USPTO_Green_fine_tuned.index))

## Now I will need to get the patent_id, patent_date, patent_title, patent_abstract, all_cpc information from 'Results/Step2/'+str(Y_category)+'_USPTO_Green_potentialGreen.csv'

df_USPTO_Green_potentialGreen_address = 'Results/Step2/' + str(Y_category) + '_USPTO_Green_potentialGreen.csv'

df_USPTO_Green_potentialGreen = pd.read_csv(df_USPTO_Green_potentialGreen_address, header=0, dtype='unicode', low_memory=False)

## Now I will merge these with df_USPTO_Green_potentialGreen to get the detailed descriptions from latter

df_USPTO_Green_fine_tuned = pd.merge(df_USPTO_Green_fine_tuned, df_USPTO_Green_potentialGreen, on='patent_id', how='left')

## I will keep only patent_id, patent_date, patent_title, patent_abstract, all_cpc

df_USPTO_Green_fine_tuned = df_USPTO_Green_fine_tuned[['patent_id', 'patent_date', 'patent_title', 'patent_abstract', 'all_CPC']]

## save it.

df_USPTO_Green_fine_tuned.to_csv('Results/Step9/' + str(Y_category) + '_USPTO_Green_fine_tuned.csv', index=False)

### Finally, I will merge these two datasets to create the Dynamic Green Dataset.

df_USPTO_Green_Dynamic = pd.concat([df_USPTO_Green_USPTO_Screened, df_USPTO_Green_fine_tuned], ignore_index=True)

print ('Dynamic Green Dataset counts')

print (len(df_USPTO_Green_Dynamic.index))

df_USPTO_Green_Dynamic.to_csv('Results/Step9/' + str(Y_category) + '_USPTO_Green_Dynamic.csv', index=False)
