#!/usr/bin/env python
# coding: utf-8

# # Training

# ## Technically, it's only a few lines of code to run on GPUs (elsewhere, ie. on Lamini).
# ```
# from llama import BasicModelRunner
# 
# model = BasicModelRunner("EleutherAI/pythia-410m") 
# model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
# model.train(is_public=True) 
# 
# 
# ```
# 1. Choose base model.
# 2. Load data.
# 3. Train it. Returns a model ID, dashboard, and playground interface.
# 
# ### Let's look under the hood at the core code running this! This is the open core of Lamini's `llama` library :)

# In[1]:


import os
import lamini

lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")


# In[2]:


import datasets
import tempfile
import logging
import random
import config
import os
import yaml
import time
import torch
import transformers
import pandas as pd
import jsonlines

from utilities import *
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from llama import BasicModelRunner


logger = logging.getLogger(__name__)
global_config = None


# ### Load the Lamini docs dataset

# In[3]:


dataset_name = "lamini_docs.jsonl"
dataset_path = f"/content/{dataset_name}"
use_hf = False


# In[4]:


dataset_path = "lamini/lamini_docs"
use_hf = True


# ### Set up the model, training config, and tokenizer

# In[5]:


model_name = "EleutherAI/pythia-70m"


# In[6]:


training_config = {
    "model": {
        "pretrained_name": model_name,
        "max_length" : 2048
    },
    "datasets": {
        "use_hf": use_hf,
        "path": dataset_path
    },
    "verbose": True
}


# In[7]:


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
train_dataset, test_dataset = tokenize_and_split_data(training_config, tokenizer)

print(train_dataset)
print(test_dataset)


# ### Load the base model

# In[8]:


base_model = AutoModelForCausalLM.from_pretrained(model_name)


# In[9]:


device_count = torch.cuda.device_count()
if device_count > 0:
    logger.debug("Select GPU device")
    device = torch.device("cuda")
else:
    logger.debug("Select CPU device")
    device = torch.device("cpu")


# In[10]:


base_model.to(device)


# ### Define function to carry out inference

# In[11]:


def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
  input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
  )

  # Generate
  device = model.device
  generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
  )

  # Decode
  generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

  # Strip the prompt
  generated_text_answer = generated_text_with_prompt[0][len(text):]

  return generated_text_answer


# ### Try the base model

# In[12]:


test_dataset[0]


# In[13]:


test_text = test_dataset[0]['question']
print(test_dataset[0]['input_ids'] == test_dataset[0]['labels'])
print("Question input (test):", test_text)
print(f"Correct answer from Lamini docs: {test_dataset[0]['answer']}")
print("Model's answer: ")
print(inference(test_text, base_model, tokenizer))


# ### Setup training

# In[14]:


max_steps = 3


# In[15]:


trained_model_name = f"lamini_docs_{max_steps}_steps"
output_dir = trained_model_name


# In[16]:


training_args = TrainingArguments(

  # Learning rate
  learning_rate=1.0e-5,

  # Number of training epochs
  num_train_epochs=1,

  # Max steps to train for (each step is a batch of data)
  # Overrides num_train_epochs, if not -1
  max_steps=max_steps,

  # Batch size for training
  per_device_train_batch_size=1,

  # Directory to save model checkpoints
  output_dir=output_dir,

  # Other arguments
  overwrite_output_dir=False, # Overwrite the content of the output directory
  disable_tqdm=False, # Disable progress bars
  eval_steps=120, # Number of update steps between two evaluations
  save_steps=120, # After # steps model is saved
  warmup_steps=1, # Number of warmup steps for learning rate scheduler
  per_device_eval_batch_size=1, # Batch size for evaluation
  evaluation_strategy="steps",
  logging_strategy="steps",
  logging_steps=1,
  optim="adafactor",
  gradient_accumulation_steps = 4,
  gradient_checkpointing=False,

  # Parameters for early stopping
  load_best_model_at_end=True,
  save_total_limit=1,
  metric_for_best_model="eval_loss",
  greater_is_better=False
)


# In[17]:


model_flops = (
  base_model.floating_point_ops(
    {
       "input_ids": torch.zeros(
           (1, training_config["model"]["max_length"])
      )
    }
  )
  * training_args.gradient_accumulation_steps
)

print(base_model)
print("Memory footprint", base_model.get_memory_footprint() / 1e9, "GB")
print("Flops", model_flops / 1e9, "GFLOPs")


# In[18]:


trainer = Trainer(
    model=base_model,
    model_flops=model_flops,
    total_steps=max_steps,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# ### Train a few steps

# In[19]:


training_output = trainer.train()


# ### Save model locally

# In[20]:


save_dir = f'{output_dir}/final'

trainer.save_model(save_dir)
print("Saved model to:", save_dir)


# In[21]:


finetuned_slightly_model = AutoModelForCausalLM.from_pretrained(save_dir, local_files_only=True)


# In[22]:


finetuned_slightly_model.to(device) 


# ### Run slightly trained model

# In[23]:


test_question = test_dataset[0]['question']
print("Question input (test):", test_question)

print("Finetuned slightly model's answer: ")
print(inference(test_question, finetuned_slightly_model, tokenizer))


# In[24]:


test_answer = test_dataset[0]['answer']
print("Target answer output (test):", test_answer)


# ### Run same model trained for two epochs 

# In[25]:


finetuned_longer_model = AutoModelForCausalLM.from_pretrained("lamini/lamini_docs_finetuned")
tokenizer = AutoTokenizer.from_pretrained("lamini/lamini_docs_finetuned")

finetuned_longer_model.to(device)
print("Finetuned longer model's answer: ")
print(inference(test_question, finetuned_longer_model, tokenizer))


# ### Run much larger trained model and explore moderation

# In[34]:


bigger_finetuned_model = BasicModelRunner("EleutherAI/pythia-410m")
bigger_finetuned_output = bigger_finetuned_model(test_question)
print("Bigger (2.8B) finetuned model (test): ", bigger_finetuned_output)


# In[35]:


count = 0
for i in range(len(train_dataset)):
 if "keep the discussion relevant to Lamini" in train_dataset[i]["answer"]:
  print(i, train_dataset[i]["question"], train_dataset[i]["answer"])
  count += 1
print(count)


# ### Explore moderation using small model
# First, try the non-finetuned base model:

# In[28]:


base_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
print(inference("What do you think of Mars?", base_model, base_tokenizer))


# ### Now try moderation with finetuned small model 

# In[29]:


print(inference("What do you think of Mars?", finetuned_longer_model, tokenizer))


# ### Finetune a model in 3 lines of code using Lamini

# In[30]:


model = BasicModelRunner("EleutherAI/pythia-410m") 
model.load_data_from_jsonlines("lamini_docs.jsonl", input_key="question", output_key="answer")
model.train(is_public=True) 


# In[31]:


out = model.evaluate()


# In[32]:


lofd = []
for e in out['eval_results']:
    q  = f"{e['input']}"
    at = f"{e['outputs'][0]['output']}"
    ab = f"{e['outputs'][1]['output']}"
    di = {'question': q, 'trained model': at, 'Base Model' : ab}
    lofd.append(di)
df = pd.DataFrame.from_dict(lofd)
style_df = df.style.set_properties(**{'text-align': 'left'})
style_df = style_df.set_properties(**{"vertical-align": "text-top"})
style_df

