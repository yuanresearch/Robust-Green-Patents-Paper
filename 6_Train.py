
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


# Input_trim = True
# score_screen_set = True
#
# df_USPTO_Green_address = 'Results/Step3/' + str(Y_category) + '_USPTO_Green.csv'
#
# df_USPTO_Green_withDetailedDescriptions = pd.read_csv(df_USPTO_Green_address, header=0, dtype='unicode', low_memory=False)
#
#
# df_USPTO_Green_address = 'Results/Step4/' + str(Y_category) + '_USPTO_Green.csv'
#
# df_USPTO_Green_with_GPT4_Response = pd.read_csv(df_USPTO_Green_address, header=0, dtype='unicode', low_memory=False)
#
# # Extract information using regular expressions and assign to new columns
# df_USPTO_Green_with_GPT4_Response['Decision'] = df_USPTO_Green_with_GPT4_Response['gpt4_is_green_patent'].str.extract(r'Decision:\s*(\w+)')
# df_USPTO_Green_with_GPT4_Response['Confidence Score'] = df_USPTO_Green_with_GPT4_Response['gpt4_is_green_patent'].str.extract(r'Confidence Score:\s*(\d+)')
# df_USPTO_Green_with_GPT4_Response['Explanation'] = df_USPTO_Green_with_GPT4_Response['gpt4_is_green_patent'].str.extract(r'Explanation:\s*(.*)', expand=False)
#
#
# ## Merge df_USPTO_Green_withDetailedDescriptions's description_text only with df_USPTO_Green_with_GPT4_Response by patent_id
#
# df_USPTO_Green_TrainDataPrepare = pd.merge(df_USPTO_Green_with_GPT4_Response[['patent_id', 'gpt4_is_green_patent', 'Decision', 'Confidence Score', 'Explanation','system_content']],
#                                            df_USPTO_Green_withDetailedDescriptions, how='left', left_on='patent_id', right_on='patent_id')
#
# ## print the column names of df_USPTO_Green_TrainDataPrepare
#
# print(df_USPTO_Green_TrainDataPrepare.columns)
#
# print (len(df_USPTO_Green_TrainDataPrepare.index))
#
# print (df_USPTO_Green_TrainDataPrepare.head(1))
#
#
# print ('Before score_screen_set!')
# print (len(df_USPTO_Green_TrainDataPrepare.index))
#
# ## some important adjustments.
# if score_screen_set:  ## I only need patents with confidence score > 95 when Decision is Yes, and confidence score > 80 when Decision is No.
#
#     df_USPTO_Green_TrainDataPrepare['Confidence Score'] = df_USPTO_Green_TrainDataPrepare['Confidence Score'].astype(float)
#
#     df_USPTO_Green_TrainDataPrepare = df_USPTO_Green_TrainDataPrepare[(df_USPTO_Green_TrainDataPrepare['Decision'] == 'Yes') & (df_USPTO_Green_TrainDataPrepare['Confidence Score'] > 95)
#                                                                       | (df_USPTO_Green_TrainDataPrepare['Decision'] == 'No') & (df_USPTO_Green_TrainDataPrepare['Confidence Score'] >= 80)]
#
# ## check how many rows left, for decision Yes, and for decision No.
#
# print ('After score_screen_set!')
#
# print ('decision Yes!')
#
# print (len(df_USPTO_Green_TrainDataPrepare[df_USPTO_Green_TrainDataPrepare['Decision'] == 'Yes'].index))
#
# print ('decision No!')
#
# print (len(df_USPTO_Green_TrainDataPrepare[df_USPTO_Green_TrainDataPrepare['Decision'] == 'No'].index))
#
# print ('decision Yes and No!')
#
# print (len(df_USPTO_Green_TrainDataPrepare.index))
#
# ## I want to sample from dicision yes rows, so the row counts from yes are exactly same as decision no.
#


# if Input_trim: ## keep the first 50 words for system contents, and description_text. I won't use the input to fine-turn the model, so it would save lots of memory.
#
#     df_USPTO_Green_TrainDataPrepare['system_content'] = df_USPTO_Green_TrainDataPrepare['system_content'].str.split().str[:50].str.join(' ')
#     df_USPTO_Green_TrainDataPrepare['description_text'] = df_USPTO_Green_TrainDataPrepare['description_text'].str.split().str[:50].str.join(' ')


### Real Start.


df_USPTO_Green_step5_address = 'Results/Step5/' + str(Y_category) + '_USPTO_Green_trainReady.csv'

df_USPTO_Green_with_GPT4_trainReady = pd.read_csv(df_USPTO_Green_step5_address, header=0, dtype='unicode', low_memory=False)

# Function to format each row as a ShareGPT-style conversation
def format_conversation(row):
    return [
        {"from": "human", "value": row["system_content"]},
        # {"from": "user", "value": row["description_text"]},
        {"from": "gpt", "value": row["gpt4_is_green_patent"]}
    ]

# Apply the function to each row and store the result in a new column
df_USPTO_Green_with_GPT4_trainReady["sharegpt_format"] = df_USPTO_Green_with_GPT4_trainReady.apply(format_conversation, axis=1)

# Display the dataframe with the new ShareGPT-style conversation
print (df_USPTO_Green_with_GPT4_trainReady[["sharegpt_format"]].head())

df_USPTO_Green_with_GPT4_trainReady.to_csv('Results/Step6/' + str(Y_category) + '_USPTO_Green_TrainDataPrepare.csv', index=False)

## drop all columns but sharegpt_format

## rename sharegpt_format to conversation

df_USPTO_Green_TrainDataPrepare = df_USPTO_Green_with_GPT4_trainReady[['sharegpt_format']]

##rename sharegpt_format to conversation

df_USPTO_Green_TrainDataPrepare = df_USPTO_Green_TrainDataPrepare.rename(columns={'sharegpt_format': 'conversations'})

## convert it into huggingface format

df_USPTO_Green_TrainDataPrepare = Dataset.from_pandas(df_USPTO_Green_TrainDataPrepare)

print (df_USPTO_Green_TrainDataPrepare[3])




################################# Training with unsloth######################################################################################
#############################################################################################################################################

import torch

print (torch.cuda.is_available())


from unsloth import FastLanguageModel
import torch
max_seq_length = 128000 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
pass

from datasets import load_dataset
# dataset = load_dataset("mlabonne/FineTome-100k", split = "train")  ## sample dataset. I switch to the green patent dataset in this research.

dataset = df_USPTO_Green_TrainDataPrepare
print (dataset[3])

from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

print (dataset[3])

print ('conversations')

print (dataset[3]["conversations"])

print ('text')

print (dataset[3]["text"])


######################
## Train the model
#####################

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

tokenizer.decode(trainer.train_dataset[3]["input_ids"])

print ('A!')

print (tokenizer.decode(trainer.train_dataset[3]["input_ids"]))

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[3]["labels"]])

print ('B!')

print (tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[3]["labels"]]))




#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# Inference

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
tokenizer.batch_decode(outputs)

print (tokenizer.batch_decode(outputs))

## Save the model locally

model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


