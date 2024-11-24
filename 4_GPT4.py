
#### Importing necessary packages.
import sys
import os
import zipfile as zip
from time import sleep

import pandas as pd
import csv
import numpy as np

import openai

# # run it from command line to downnload detail description text files.
# for year in {1976..2024}; do
#     wget https://s3.amazonaws.com/data.patentsview.org/detail-description-text/g_detail_desc_text_${year}.tsv.zip &&
#     unzip g_detail_desc_text_${year}.tsv.zip &&
#     rm g_detail_desc_text_${year}.tsv.zip
# done

## For the training data and candidate dataset, we need descriptions from the raw files.


import pandas as pd
import os
import time


Y_category ='Y02T'

os.environ["http_proxy"] = "http://localhost:8890"
os.environ["https_proxy"] = "http://localhost:8890"

openai.api_key= ''


# Read df_USPTO_Green and df_USPTO_Green_potentialGreen

df_USPTO_Green_address = 'Results/Step3/' + str(Y_category) + '_USPTO_Green.csv'

df_USPTO_Green = pd.read_csv(df_USPTO_Green_address, header=0, dtype='unicode', low_memory=False)


## Print the column names of df_USPTO_Green and df_USPTO_Green_potentialGreen

print(df_USPTO_Green.columns)

##I need to check whether there are any missing values in the patent_id column of df_USPTO_Green in the description_text column.
##If it does, drop the row. firt print the original number of rows in the dataframes.

print(len(df_USPTO_Green.index))

## Drop rows with missing values in the description_text column of df_USPTO_Green

df_USPTO_Green = df_USPTO_Green.dropna(subset=['description_text'])

print(len(df_USPTO_Green.index))

# ### Total token count for the entire df_USPTO_Green y02t: 929146553.
#
# import tiktoken
#
# def calculate_total_token_count(df, model="gpt-4o-mini"):
#     # Use tiktoken to calculate total token count for the description_text column
#     encoding = tiktoken.encoding_for_model(model)
#     total_tokens = sum(len(encoding.encode(text)) for text in df['description_text'])
#     return total_tokens
#
# # Calculate the total token count for the entire DataFrame
# total_token_count = calculate_total_token_count(df_USPTO_Green)
# print(f"Total token count for the entire df_USPTO_Green: {total_token_count}")

# Error : This model's maximum context length is 128000 tokens. However, your messages resulted in 128216 tokens. Please reduce the length of the messages.



### Now I am classifying all USPTO-defined green patent, and trying to get negative training data if the return is not green.

# data_df_USPTO_Green = df_USPTO_Green.head(1000).copy()
data_df_USPTO_Green = df_USPTO_Green.copy()

data_df_USPTO_Green = data_df_USPTO_Green.reset_index(drop=True)

#data_df_USPTO_Green = df_USPTO_Green.copy()

def add_predictions_to_dataframe(df):
    # Ensure 'description_text' column exists before proceeding
    if 'description_text' not in df.columns:
        print("Error: 'description_text' column not found in the DataFrame")
        return df

    # Iterate through each row
    for index in range(len(df)):
        row = df.iloc[index]
        patent_content = row['description_text']

        #system_content = 'You are a great green patent examiner. Please tell me whether the following green patent description suggests this patent is green or not. Answer Yes or  No only.'
        # system_content = ('You are an expert green patent examiner. I will provide you with a detailed description of a patent. '
        #                   'Your task is to assess whether it qualifies as a "green patent," which refers to patents focused on environmentally sustainable innovations. '
        #                   'Please respond with the following format:  '
        #                   '1.Decision: State "Yes" if it qualifies as a green patent or "No" if it does not. '
        #                   '2.Confidence Score: Provide a confidence score between 1 and 100 to indicate your confidence level in the accuracy of the "Yes" or "No" decision. For example, a score close to 100 suggests high confidence in the provided response. '
        #                   '3.Explanation: In 200 words or less, explain the reason for your decision, addressing key aspects of the patent description that influenced your assessment. '
        #                   'Here is the patent description:')


        # system_content = ('You are an expert green patent examiner. I will provide you with a detailed description of a patent. '
        #                   'Your task is to assess whether it qualifies as a "green patent," which refers to patents focused on environmentally sustainable innovations. '
        #                   'Please respond in the exact format below, paying special attention to the "Explanation" section. '
        #                   'In "Explanation," provide a clear, logical breakdown of your decision, highlighting specific environmental benefits or lack thereof, any innovative sustainability features, and how well the patent aligns with green patent standards. Your goal is to provide a highly detailed, self-contained explanation that would be suitable as input for fine-tuning other expert models. '
        #                   '1.Decision: Yes or No (State "Yes" if it qualifies as a green patent or "No" if it does not.) '
        #                   '2.Confidence Score: Provide a confidence score between 1 and 100 to indicate your confidence level in the accuracy of the "Yes" or "No" decision. A score close to 100 suggests high confidence in the response. '
        #                   '3.Explanation: Write a detailed explanation in 200 words or less that clearly summarizes the core environmental benefits or shortcomings of the patent. Describe any unique or innovative features that contribute to its sustainability, and discuss how well the patent aligns with the established standards and definitions of a green patent. This explanation should be thorough, allowing it to serve as high-quality input data to train another model to make similar assessments independently. '
        #                   'Here is the patent description:')

        system_content = ('You are an expert green patent examiner. Your task is to assess whether the subject qualifies as a "green patent," which refers to patents focused on environmentally sustainable innovations. '
                          'Please respond strictly in the following format: '
                          '1.Decision: Yes or No (State "Yes" if it qualifies as a green patent or "No" if it does not.) '
                          '2.Confidence Score: Provide a confidence score between 1 and 100 to indicate your confidence level in the accuracy of the "Yes" or "No" decision.')



        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
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

        # Save the progress to a partial file without unnecessary columns
        if index % 10000 == 0 or index == len(df) - 1:  # Save progress every 100 rows or at the end
            df_partial = df.drop(columns=['description_text', 'description_length'], errors='ignore').copy()
            df_partial.to_csv('Results/Step4/partial_' + str(Y_category) + '_USPTO_Green.csv', index=False)
            
        # Add a 10-minute delay every 5000 iterations
        if index % 10000 == 0 and index != 0:
            print("Sleeping for 10 minutes to avoid rate limiting...")
            time.sleep(600)

    return df

# Add predictions to the DataFrame
data_df_USPTO_Green = add_predictions_to_dataframe(data_df_USPTO_Green)

# Drop unnecessary columns to save disk space
data_df_USPTO_Green = data_df_USPTO_Green.drop(columns=['description_text', 'description_length'], errors='ignore').copy()

# Save the updated DataFrame to a new CSV file
data_df_USPTO_Green.to_csv('Results/Step4/' + str(Y_category) + '_USPTO_Green.csv', index=False)

# Remove the partial file after successful completion
if os.path.exists('Results/Step4/partial_' + str(Y_category) + '_USPTO_Green.csv'):
    os.remove('Results/Step4/partial_' + str(Y_category) + '_USPTO_Green.csv')

print("Processing complete. Saved the file with green patent information.")



















