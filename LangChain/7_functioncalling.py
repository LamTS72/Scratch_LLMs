#!/usr/bin/env python
# coding: utf-8

# # OpenAI Function Calling
# 

# **Notes**:
# - LLM's don't always produce the same results. The results you see in this notebook may differ from the results you see in the video.
# - Notebooks results are temporary. Download the notebooks to your local machine if you wish to save your results.

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


import json

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_info = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


# In[3]:


# define a function
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


# In[4]:


messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston?"
    }
]


# In[5]:


import openai


# In[7]:


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions
)


# In[8]:


print(response)


# In[9]:


response_message = response["choices"][0]["message"]


# In[10]:


response_message


# In[11]:


response_message["content"]


# In[12]:


response_message["function_call"]


# In[13]:


json.loads(response_message["function_call"]["arguments"])


# In[14]:


args = json.loads(response_message["function_call"]["arguments"])


# In[15]:


get_current_weather(args)


# In[16]:


messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]


# In[18]:


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
)


# In[19]:


print(response)


# In[21]:


messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="auto",
)
print(response)


# In[22]:


messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)


# In[25]:


messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call="none",
)
print(response)


# In[27]:


messages = [
    {
        "role": "user",
        "content": "hi!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)


# In[29]:


messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston!",
    }
]
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    functions=functions,
    function_call={"name": "get_current_weather"},
)
print(response)


# In[30]:


messages.append(response["choices"][0]["message"])


# In[31]:


args = json.loads(response["choices"][0]["message"]['function_call']['arguments'])
observation = get_current_weather(args)


# In[32]:


messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": observation,
        }
)


# In[34]:


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
)
print(response)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




