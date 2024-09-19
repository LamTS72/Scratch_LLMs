#!/usr/bin/env python
# coding: utf-8

# # Lecture 2: Data Preparation
# 
# In this lesson you'll carry out some of the data cleaning steps required to prepare data for pretraining. In the video, Sung mentioned an Upstage tool called **Dataverse** which can help you with data cleaning. You can checkout the features of Dataverse at [this link](https://github.com/UpstageAI/dataverse).

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# ## 1. Sourcing datasets for pretraining
# 
# In this section, you'll see two ways to source data for training:
# 1. Download an existing dataset from Hugging Face
# 2. Create a dataset of python scripts sourced from Github
# 
# In both cases the result will be a Hugging Face `Dataset` object, part of the `Datasets` library. You can read more about the properties of Datasets and how to work with them on the [Hugging Face website](https://huggingface.co/docs/datasets/en/index).
# 
# ### Download data from Hugging face
# 
# The dataset you download here is a subset of a much larger dataset called **Red Pajama**. The full, 1 trillion token dataset is available on Hugging Face at [this link](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T).

# In[2]:


import datasets
pretraining_dataset = datasets.load_dataset(
    "upstage/Pretraining_Dataset",
    split="train"
)


# In[3]:


print(pretraining_dataset)


# Only work with the `text` column:

# In[4]:


pretraining_dataset = pretraining_dataset.select_columns(
    ["text"]
)


# Print a sample:

# In[5]:


print(pretraining_dataset[0]["text"][:500])


# ### Compare pretraining and fine-tuning datasets
# In the next cell, you'll download a fine-tuning dataset to contrast with the pretraining dataset you loaded above. You can read more about the Alpaca model and instruction tuning dataset [here](https://crfm.stanford.edu/2023/03/13/alpaca.html). 

# In[6]:


instruction_dataset = datasets.load_dataset(
    "c-s-ale/alpaca-gpt4-data",
    split='train'
)
print(instruction_dataset)


# In[7]:


i=0
print("Instruction: " + instruction_dataset[i]["instruction"] 
      + "\nInput: " + instruction_dataset[i]["input"] 
      + "\nOutput: " + instruction_dataset[i]["output"])


# Notice how in contrast to the pretraining data, which is just raw text, fine-tuning datasets are structured into question-answer pairs or instruction-response sets that can include additional input context if required. 
# 
# Moving forward, you'll only work with the unstructured pretraining dataset.

# ### Scrape python code from Github
# Here, you'll download a selection of python scripts from Github and then prepare them as a Hugging Face `Dataset` object to use in training. 
# 
# The same pattern here will work for preparing any text scraped from the web.

# In[8]:


# Import some required packages
import os
import requests

# Path to directory to store python scripts
code_dir = "./code"


# In[9]:


urls = [
    "https://raw.githubusercontent.com/TheAlgorithms/Python/master/searches/double_linear_search_recursion.py",
    "https://raw.githubusercontent.com/KosingZhu/tensorflow/master/tensorflow/python/tools/module_util.py",
    "https://raw.githubusercontent.com/EricRemmerswaal/tensorflow/master/tensorflow/python/distribute/distribute_coordinator_context.py",
    "https://raw.githubusercontent.com/computationalartist/tensorflow/master/tensorflow/python/ops/numpy_ops/integration_test/benchmarks/numpy_mlp.py",
    "https://raw.githubusercontent.com/Van-an/tensorflow/master/tensorflow/python/distribute/coordinator/values.py",
    "https://raw.githubusercontent.com/nkgwer/tensorflow/master/tensorflow/lite/tools/visualize.py",
    "https://raw.githubusercontent.com/gitblazer/youtube-dl/master/youtube_dl/version.py",
    "https://raw.githubusercontent.com/Joshua-Barawa/My-Photos/master/venv/lib/python3.8/site-packages/django/contrib/messages/__init__.py",
    "https://raw.githubusercontent.com/PaliC/pytorch/master/test/fx/test_subgraph_rewriter.py"
]


# Retrieve the python scripts:

# In[10]:


for url in urls:
    print(f"Working on url: {url}")
    response = requests.get(url)
    file_name = os.path.basename(url)
    file_path = os.path.join(code_dir, file_name)
    
    with open(file_path, "wb") as file:
        file.write(response.content)


# In[11]:


files = os.listdir(code_dir)
for file in files:
    print(file)


# Concatenate scripts into a list:

# In[12]:


code_dataset = []
for file in os.listdir(code_dir):
    code_dataset.append(
        {'text': open(os.path.join(code_dir, file), 'r').read()}
    )


# Convert list to Hugging Face `Dataset` object:

# In[13]:


code_dataset = datasets.Dataset.from_list(code_dataset)
print(code_dataset)


# Combine the python code dataset with the pretraining dataset you downloaded above:

# In[14]:


dataset = datasets.concatenate_datasets(
    [pretraining_dataset, code_dataset]
)
print(dataset)


# ## 2. Data cleaning
# 
# In the cells below, you'll carry out the following cleaning steps:
# 1. Filter out samples that are too short
# 2. Remove repetitions within a single text example
# 3. Remove duplicated documents
# 4. Quality filter to remove non-English texts 

# In[15]:


dataset.num_rows


# ### Remove examples that are too short

# In[16]:


import heapq

def paragraph_length_filter(x):
    """Returns False iff a page has too few lines or lines are too short."""
    lines = x['text'].split('\n')
    if (
        len(lines) < 3
        or min(heapq.nlargest(3, [len(line) for line in lines])) < 3
    ):
        return False
    return True


# In[17]:


dataset = dataset.filter(
    paragraph_length_filter,
    load_from_cache_file=False
)


# In[18]:


dataset.num_rows


# ### Remove repeated text within training examples
# 
# Here you'll remove text repetitions within each example. 

# In[19]:


def find_duplicates(paragraphs):
    """
    Use this function to find the number of repetitions 
    in the paragraphs.
    """
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in paragraphs:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1
        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars


# In[20]:


import re

def paragraph_repetition_filter(x):
    """
    Returns False iff a page has too many repetitions.
    """
    text = x['text']
    paragraphs = re.compile(r"\n{2,}").split(text.strip())                # Split by paragraphs (2 or more newlines)
    paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)  # Find number of duplicates in paragraphs
    if paragraphs_duplicates / len(paragraphs) > 0.3:
        return False
    if char_duplicates / len(text) > 0.2:
        return False
    return True


# In[21]:


dataset = dataset.filter(
    paragraph_repetition_filter,
    load_from_cache_file=False
)


# In[22]:


dataset.num_rows


# ### Deduplication
# 
# In this section, you'll remove duplicate examples from the entire dataset (in contrast to the previous step where you were just looking for repeated text in each example.)

# In[23]:


def deduplication(ds):
    def dedup_func(x):
        """Use this function to remove duplicate entries"""
        if x['text'] in unique_text:
            return False
        else:
            unique_text.add(x['text'])
            return True

    unique_text = set()

    ds = ds.filter(dedup_func, load_from_cache_file=False, num_proc=1)
    return ds

dataset = deduplication(dataset)


# In[24]:


dataset.num_rows


# ### Quality filter - Language
# 
# Here you'll remove any text examples that are in a language other than English. The code here uses a language detection model called fastText. You can read about fastText [here](https://fasttext.cc/).

# In[25]:


# !pip install fasttext


# In[26]:


import urllib
from fasttext.FastText import _FastText

def english_language_filter(ds):
    # load language detection model
    model = _FastText('./models/upstage/L2_language_model.bin')
    
    def is_english(x):
        # Predict language of the text and probability
        language, score = model.predict(x['text'].replace("\n", ""))

        language = language[0].split("__")[2]
        return score > 0.4 and language == "en" # change code here if building a model in another language

    ds = ds.filter(is_english, load_from_cache_file=False, num_proc=1)
    return ds

dataset = english_language_filter(dataset)


# In[27]:


dataset.num_rows


# ## 3. Save the dataset to disk
# 
# Read more about the parquet data format [here](https://parquet.apache.org/).

# In[ ]:


file_path = "./data/preprocessed_dataset.parquet"
dataset.to_parquet(file_path)


# In[ ]:





# In[ ]:





# In[ ]:




