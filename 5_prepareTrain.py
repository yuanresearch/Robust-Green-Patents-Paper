
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
from sqlalchemy.sql.operators import truediv

Y_category ='Y02T'
os.environ["http_proxy"] = "http://localhost:8890"
os.environ["https_proxy"] = "http://localhost:8890"

openai.api_key= ''


Input_trim = True
score_screen_set = True

df_USPTO_Green_step3_address = 'Results/Step3/' + str(Y_category) + '_USPTO_Green.csv'

df_USPTO_Green_withDetailedDescriptions = pd.read_csv(df_USPTO_Green_step3_address, header=0, dtype='unicode', low_memory=False)


df_USPTO_Green_step4_address = 'Results/Step4/' + str(Y_category) + '_USPTO_Green.csv'

df_USPTO_Green_with_GPT4_Response = pd.read_csv(df_USPTO_Green_step4_address, header=0, dtype='unicode', low_memory=False)


### I will need to construct a new dataframe
### with df_USPTO_Green_with_GPT4_Response when Decision is NO, and confidence score is above 85, along with the patents defined by data_df_USPTO_Green_expert_positive.

# Extract information using regular expressions and assign to new columns
df_USPTO_Green_with_GPT4_Response['Decision'] = df_USPTO_Green_with_GPT4_Response['gpt4_is_green_patent'].str.extract(r'Decision:\s*(\w+)')
df_USPTO_Green_with_GPT4_Response['Confidence Score'] = df_USPTO_Green_with_GPT4_Response['gpt4_is_green_patent'].str.extract(r'Confidence Score:\s*(\d+)')

df_USPTO_Green_with_GPT4_Response['Confidence Score'] = df_USPTO_Green_with_GPT4_Response['Confidence Score'].astype(float)

df_USPTO_Green_with_GPT4_Response_screened = df_USPTO_Green_with_GPT4_Response[(df_USPTO_Green_with_GPT4_Response['Decision'] == 'No') & (
                df_USPTO_Green_with_GPT4_Response['Confidence Score'] >= 85)].copy()  ### Negative training data

print ('df_USPTO_Green_with_GPT4_Response, screened by confidence score and is it recognized as non-green patent')

print (len(df_USPTO_Green_with_GPT4_Response_screened.index))

## print their patent_id

#print (df_USPTO_Green_with_GPT4_Response_screened['patent_id'].tolist())


### Now I will get the patent data from df_USPTO_Green_withDetailedDescriptions based on patent_id for both positive and negative training data.

data_df_USPTO_Green_expert_positive = df_USPTO_Green_withDetailedDescriptions[df_USPTO_Green_withDetailedDescriptions['CPC0'].isin(['C10L', 'B29D', 'F25B', 'G08G', 'H05B'])].copy()  ## Positive training data

print (data_df_USPTO_Green_expert_positive['CPC0'].value_counts())

data_df_USPTO_Green_expert_negative = df_USPTO_Green_withDetailedDescriptions[df_USPTO_Green_withDetailedDescriptions['patent_id'].isin(df_USPTO_Green_with_GPT4_Response_screened['patent_id'].tolist())].copy()  ## Negative training data

print (data_df_USPTO_Green_expert_negative['CPC0'].value_counts())

## I am creating a new column between 0 or 1, where 1 is positive training data, and 0 is negative training data.

data_df_USPTO_Green_expert_positive['label'] = 1

data_df_USPTO_Green_expert_negative['label'] = 0

## Now I will merge the positive and negative training data.

data_df_USPTO_Green_training = pd.concat([data_df_USPTO_Green_expert_positive, data_df_USPTO_Green_expert_negative], ignore_index=True)

print (data_df_USPTO_Green_training['label'].value_counts())

## Now I wll keep same counts of positive and negative training data, if the counts are not equal, I will randomly select the same number of negative training data. Vice versa.

if len(data_df_USPTO_Green_expert_positive.index) > len(data_df_USPTO_Green_expert_negative.index):

    data_df_USPTO_Green_training = data_df_USPTO_Green_training.sample(frac=1).groupby('label').head(len(data_df_USPTO_Green_expert_negative.index)).reset_index(drop=True)

elif len(data_df_USPTO_Green_expert_positive.index) < len(data_df_USPTO_Green_expert_negative.index):

    data_df_USPTO_Green_training = data_df_USPTO_Green_training.sample(frac=1).groupby('label').head(len(data_df_USPTO_Green_expert_positive.index)*2).reset_index(drop=True)

pd.set_option('display.max_rows', 500)

print (data_df_USPTO_Green_training['label'].value_counts())

print (data_df_USPTO_Green_training['CPC0'].value_counts())

##############################################################################################

## Now I will use GPT4o (this is the best, but very expensive!!!) instead of mini, to get the most advanced detailed explanation of why GPT think the patent is green or not green.


data_df_USPTO_Green_training = data_df_USPTO_Green_training.reset_index(drop=True)


# data_df_USPTO_Green = df_USPTO_Green.copy()

def add_predictions_to_dataframe(df):
    # Ensure 'description_text' column exists before proceeding
    if 'description_text' not in df.columns:
        print("Error: 'description_text' column not found in the DataFrame")
        return df

    # Iterate through each row
    for index in range(len(df)):
        row = df.iloc[index]
        patent_content = row['description_text']

        system_content = (
            'You are a great green patent examiner, and determine if it qualifies as a green patent. Begin with a Yes, this is a green patent or No, this is not a green patent to indicate if it meets green patent criteria. '
            'Then, in a summary of 200 to 300 words, identify the core environmental benefits or shortcomings of the patent. Highlight any unique or innovative features contributing to its sustainability, especially those aligned with established green patent standards.'
            'Ensure the explanation is concise and includes key markers to aid in distinguishing green patents from non-green patents.')

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": patent_content}
                ]
            )

            # Extract the last message from the response
            last_message = response.choices[0].message['content']  # Get the content of the last message
            df.at[index, 'gpt4_is_green_patent'] = last_message
            df.at[index, 'system_content'] = system_content

            print(f"Processed index: {index}")
            print(last_message)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            df.at[index, 'gpt4_is_green_patent'] = 'error'


    return df


# Add predictions to the DataFrame
data_df_USPTO_Green_training = add_predictions_to_dataframe(data_df_USPTO_Green_training)

# Drop unnecessary columns to save disk space
data_df_USPTO_Green_training = data_df_USPTO_Green_training.drop(columns=['description_text', 'description_length'],
                                               errors='ignore').copy()

# Save the updated DataFrame to a new CSV file
data_df_USPTO_Green_training.to_csv('Results/Step5/' + str(Y_category) + '_USPTO_Green_trainReady.csv', index=False)

