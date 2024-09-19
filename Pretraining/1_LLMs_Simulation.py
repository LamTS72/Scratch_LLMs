#!/usr/bin/env python
# coding: utf-8

# # Lesson 1: Why Pretraining?
# 
# ## 1. Install dependencies and fix seed
# 
# Welcome to Lesson 1!
# 
# If you would like to access the `requirements.txt` file for this course, go to `File` and click on `Open`.

# In[1]:


# Install any packages if it does not exist
# !pip install -q -r ../requirements.txt


# In[2]:


# Ignore insignificant warnings (ex: deprecations)
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Set a seed for reproducibility
import torch

def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()


# ## 2. Load a general pretrained model
# 
# This course will work with small models that fit within the memory of the learning platform. TinySolar-248m-4k is a small decoder-only model with 248M parameters (similar in scale to GPT2) and a 4096 token context window. You can find the model on the Hugging Face model library at [this link](https://huggingface.co/upstage/TinySolar-248m-4k).
# 
# You'll load the model in three steps:
# 1. Specify the path to the model in the Hugging Face model library
# 2. Load the model using `AutoModelforCausalLM` in the `transformers` library
# 3. Load the tokenizer for the model from the same model path

# In[4]:


model_path_or_name = "./models/TinySolar-248m-4k"


# In[5]:


from transformers import AutoModelForCausalLM
tiny_general_model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    device_map="cpu", # change to auto if you have access to a GPU
    torch_dtype=torch.bfloat16
)


# In[6]:


from transformers import AutoTokenizer
tiny_general_tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name
)


# ## 3. Generate text samples
# 
# Here you'll try generating some text with the model. You'll set a prompt, instantiate a text streamer, and then have the model complete the prompt:

# In[7]:


prompt = "I am an engineer. I love"


# In[8]:


inputs = tiny_general_tokenizer(prompt, return_tensors="pt")


# In[9]:


from transformers import TextStreamer
streamer = TextStreamer(
    tiny_general_tokenizer,
    skip_prompt=True, # If you set to false, the model will first return the prompt and then the generated text
    skip_special_tokens=True
)


# In[10]:


outputs = tiny_general_model.generate(
    **inputs, 
    streamer=streamer, 
    use_cache=True,
    max_new_tokens=128,
    do_sample=False, 
    temperature=0.0,
    repetition_penalty=1.1
)


# ## 4. Generate Python samples with pretrained general model
# 
# Use the model to write a python function called `find_max()` that finds the maximum value in a list of numbers:

# In[11]:


prompt =  "def find_max(numbers):"


# In[12]:


inputs = tiny_general_tokenizer(
    prompt, return_tensors="pt"
).to(tiny_general_model.device)

streamer = TextStreamer(
    tiny_general_tokenizer, 
    skip_prompt=True, # Set to false to include the prompt in the output
    skip_special_tokens=True
)


# In[13]:


outputs = tiny_general_model.generate(
    **inputs, 
    streamer=streamer, 
    use_cache=True, 
    max_new_tokens=128, 
    do_sample=False, 
    temperature=0.0, 
    repetition_penalty=1.1
)


# ## 5. Generate Python samples with finetuned Python model
# 
# This model has been fine-tuned on instruction code examples. You can find the model and information about the fine-tuning datasets on the Hugging Face model library at [this link](https://huggingface.co/upstage/TinySolar-248m-4k-code-instruct).
# 
# You'll follow the same steps as above to load the model and use it to generate text.

# In[14]:


model_path_or_name = "./models/TinySolar-248m-4k-code-instruct"


# In[15]:


tiny_finetuned_model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

tiny_finetuned_tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name
)


# In[16]:


prompt =  "def find_max(numbers):"

inputs = tiny_finetuned_tokenizer(
    prompt, return_tensors="pt"
).to(tiny_finetuned_model.device)

streamer = TextStreamer(
    tiny_finetuned_tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

outputs = tiny_finetuned_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.1
)


# ## 6. Generate Python samples with pretrained Python model
# 
# Here you'll use a version of TinySolar-248m-4k that has been further pretrained (a process called **continued pretraining**) on a large selection of python code samples. You can find the model on Hugging Face at [this link](https://huggingface.co/upstage/TinySolar-248m-4k-py).
# 
# You'll follow the same steps as above to load the model and use it to generate text.

# In[17]:


model_path_or_name = "./models/TinySolar-248m-4k-py" 


# In[18]:


tiny_custom_model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    device_map="cpu",
    torch_dtype=torch.bfloat16,    
)

tiny_custom_tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name
)


# In[19]:


prompt = "def find_max(numbers):"

inputs = tiny_custom_tokenizer(
    prompt, return_tensors="pt"
).to(tiny_custom_model.device)

streamer = TextStreamer(
    tiny_custom_tokenizer,
    skip_prompt=True, 
    skip_special_tokens=True
)

outputs = tiny_custom_model.generate(
    **inputs, streamer=streamer,
    use_cache=True, 
    max_new_tokens=128, 
    do_sample=False, 
    repetition_penalty=1.1
)


# Try running the python code the model generated above:

# In[20]:


def find_max(numbers):
   max = 0
   for num in numbers:
       if num > max:
           max = num
   return max


# In[21]:


find_max([1,3,5,1,6,7,2])

