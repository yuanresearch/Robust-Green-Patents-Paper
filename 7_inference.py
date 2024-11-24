
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


#######################################################################
import torch
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Check if CUDA is available
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please check your GPU settings.")

# Load the dataset
df_USPTO_Green_address = 'Results/Step3/' + str(Y_category) + '_USPTO_Green_potentialGreen.csv'
df_USPTO_Green_potentialGreen = pd.read_csv(df_USPTO_Green_address, header=0, dtype='unicode', low_memory=False)

# Load the model and tokenizer once
max_seq_length = 128000  # Adjust as needed
dtype = None  # Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# System prompt for the model
system_content = (
    'You are an expert green patent examiner. I will provide a detailed description of a patent. temperature = 0.0'
    'Your task is to assess whether this patent qualifies as a "green patents" related to transportation.'
    'Please evaluate it based on this criterion.'
    'Output the response strictly in the following format without explanations:'
    'Decision: Yes or No (Answer "Yes" if it qualifies as a green patent, otherwise "No.")'
    'Confidence Score: (Provide a confidence score from 1 to 100, indicating your certainty in this decision. A score near 100 reflects high confidence.)'
    'Here is the detailed description of the patent:'
    # 'You are an expert green patent examiner. I will provide you with a detailed description of a patent. '
    # 'Your task is to assess whether it qualifies as a "green patent," which refers to patents focused on environmentally sustainable innovations. '
    # 'Please respond in the strict format below. You do not provide any explanations in this task. '
    # '1.Decision: Yes or No (State "Yes" if it qualifies as a green patent or "No" if it does not.) '
    # '2.Confidence Score: Provide a confidence score between 1 and 100 to indicate your confidence level in the accuracy of the "Yes" or "No" decision. A score close to 100 suggests high confidence in the response.'
    # 'Here is the detailed description of the patent: '
)

# Loop through the dataset and process one by one
num_rows = min(200, len(df_USPTO_Green_potentialGreen))  ###TEST

#num_rows = len(df_USPTO_Green_potentialGreen)

print(num_rows)

inference_results = []

for index in range(num_rows):
    row = df_USPTO_Green_potentialGreen.iloc[index]
    patent_id = row['patent_id']
    print(f"Processing PatentID: {patent_id}")
    patent_content = row['description_text']

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": patent_content}
    ]

    try:
        # Tokenize the input
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        # Generate response
        outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, temperature=1.5, min_p=0.1)
        response_text = (tokenizer.batch_decode(outputs))[0]

        # Find the position of the assistant tag
        start_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        start_idx = response_text.find(start_tag) + len(start_tag)

        # Find the position of the end of text tag
        end_idx = response_text.find("<|eot_id|>", start_idx)

        # Extract the assistant's answer
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            assistant_answer = response_text[start_idx:end_idx].strip()
        else:
            print(f"Unexpected response format for PatentID: {patent_id}")
            assistant_answer = ""

        print (assistant_answer)

        # Append the result to the inference_results list
        inference_results.append({
            'patent_id': patent_id,
            'assistant_answer': assistant_answer
        })

        print(f"Processed index: {index}")

    except torch.cuda.CudaError as e:
        print(f"CUDA error: {e}. Consider lowering the batch size to prevent GPU memory issues.")
        continue
    except Exception as e:
        print(f"General error: {e}")
        continue



# Convert the results list to a DataFrame and save it as a CSV file
results_df = pd.DataFrame(inference_results)
# Save the updated DataFrame to a new CSV file
results_df.to_csv('Results/Step7/' + str(Y_category) + '_USPTO_Green_infer.csv', index=False)
print("Results saved!'")

