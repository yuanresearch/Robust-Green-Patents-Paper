
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

Y_category ='Y02T'
os.environ["http_proxy"] = "http://localhost:8890"
os.environ["https_proxy"] = "http://localhost:8890"

openai.api_key= ''

## Here I am going to do some validation from sample patents. I will compare
# 1) what patent examiner's non-green patent;
# 2) what inferred from GPT4O-mini/GPT4O LATEST model directly;
# 3) what inferred from my fine-tuned LLM model in this research;
# 4) what is the actual green patent defined by manual inspection, this is considered as ground truth.


df_USPTO_Green_fine_tuned_address = 'Results/Step7/' + str(Y_category) + '_USPTO_Green_infer.csv'

df_USPTO_Green_fine_tuned = pd.read_csv(df_USPTO_Green_fine_tuned_address, header=0, dtype='unicode', low_memory=False)


### I need to clean the assistant_answer, because self-defined LLM might return no-standard answers.
### I want to get Yes or NO the first time it showed in the assistant_answer column, and the confidence score (digits) if there is any.

import pandas as pd
import re


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
print("Results Cleaned")



####Now I will merge these with df_USPTO_Green_potentialGreen to get the detailed descriptions from latter

df_USPTO_Green_address = 'Results/Step3/' + str(Y_category) + '_USPTO_Green_potentialGreen.csv'
df_USPTO_Green_potentialGreen = pd.read_csv(df_USPTO_Green_address, header=0, dtype='unicode', low_memory=False)

df_USPTO_Green_fine_tuned_sampled = processed_df.head(100) ## sample patents

## Left merge df_USPTO_Green_fine_tuned with df_USPTO_Green_potentialGreen

df_USPTO_Green_fine_tuned_sampled = pd.merge(df_USPTO_Green_fine_tuned_sampled, df_USPTO_Green_potentialGreen, on='patent_id', how='left')

## print the counts of the sample patents

print ('Counts of the sample patents')

print (len(df_USPTO_Green_fine_tuned_sampled.index))


### Now called GPT4O-mini model and GPT4o to infer the green patent status of the sample patents,respectively

data_df_USPTO_Green = df_USPTO_Green_fine_tuned_sampled.copy()

data_df_USPTO_Green = data_df_USPTO_Green.reset_index(drop=True)


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
            'You are an expert green patent examiner. Your task is to assess whether the subject qualifies as a transportation related "green patent," which refers to patents focused on transportation environmentally sustainable innovations. '
            'Output the response strictly in the following format without explanations:'
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
            df.at[index, 'gpt_4o_mini'] = last_message


            response_gpt_4o = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": patent_content}
                ]
            )

            # Extract the last message from the response
            last_message_gpt_4o = response_gpt_4o.choices[0].message['content']  # Get the content of the last message
            df.at[index, 'gpt_4o'] = last_message_gpt_4o


            # response_o1_preview = openai.ChatCompletion.create(
            #     model="o1-preview",
            #     messages=[
            #         {"role": "system", "content": system_content},
            #         {"role": "user", "content": patent_content}
            #     ]
            # )
            #
            # # Extract the last message from the response
            # last_message_o1_preview = response_o1_preview.choices[0].message['content']  # Get the content of the last message
            # df.at[index, 'o1_preview'] = last_message_o1_preview


            print(f"Processed index: {index}")
            print(last_message)
            print(last_message_gpt_4o)
            # print(last_message_o1_preview)

        except Exception as e:
            print(f"Error at index {index}: {e}")
            df.at[index, 'gpt_4o_mini'] = 'error'
            df.at[index, 'gpt_4o'] = 'error'
            # df.at[index, 'o1_preview'] = 'error'

    return df

# Add predictions to the DataFrame
data_df_USPTO_Green = add_predictions_to_dataframe(data_df_USPTO_Green)

## Add another column named "USPTO defined" to the data_df_USPTO_Green with all values set to No

data_df_USPTO_Green['USPTO defined'] = 'No'
data_df_USPTO_Green['human'] = ''

### change assistant_answer column name to fine_tuned_LLM

data_df_USPTO_Green = data_df_USPTO_Green.rename(columns={'assistant_answer': 'fine_tuned_LLM'})

### I will keep "patent_id", "assistant_answer","patent_date",""patent_title", "gpt-4o-mini", "gpt_4o", 'USPTO defined' in data_df_USPTO_Green

data_df_USPTO_Green = data_df_USPTO_Green[['patent_id', 'patent_date', 'USPTO defined', 'gpt_4o_mini', 'gpt_4o', 'human', 'ft_llm_decision','ft_llm_confidence_score', 'patent_title','ft_llm_raw']]

### Now I will save the data_df_USPTO_Green to a csv file

data_df_USPTO_Green.to_csv('Results/Step8/' + str(Y_category) + '_USPTO_Green_potentialGreen_sample.csv', index=False)

