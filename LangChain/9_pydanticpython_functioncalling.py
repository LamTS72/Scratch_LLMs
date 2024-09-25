#!/usr/bin/env python
# coding: utf-8

# # OpenAI Function Calling In LangChain

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


from typing import List
from pydantic import BaseModel, Field


# ## Pydantic Syntax
# 
# Pydantic data classes are a blend of Python's data classes with the validation power of Pydantic. 
# 
# They offer a concise way to define data structures while ensuring that the data adheres to specified types and constraints.
# 
# In standard python you would create a class like this:

# In[3]:


class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


# In[4]:


foo = User(name="Joe",age=32, email="joe@gmail.com")


# In[5]:


foo.name


# In[6]:


foo = User(name="Joe",age="bar", email="joe@gmail.com")


# In[7]:


foo.age


# In[8]:


class pUser(BaseModel):
    name: str
    age: int
    email: str


# In[9]:


foo_p = pUser(name="Jane", age=32, email="jane@gmail.com")


# In[10]:


foo_p.name


# **Note**: The next cell is expected to fail.

# In[11]:


foo_p = pUser(name="Jane", age="bar", email="jane@gmail.com")


# In[12]:


class Class(BaseModel):
    students: List[pUser]


# In[13]:


obj = Class(
    students=[pUser(name="Jane", age=32, email="jane@gmail.com")]
)


# In[14]:


obj


# ## Pydantic to OpenAI function definition
# 

# In[15]:


class WeatherSearch(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str = Field(description="airport code to get weather for")


# In[16]:


from langchain.utils.openai_functions import convert_pydantic_to_openai_function


# In[17]:


weather_function = convert_pydantic_to_openai_function(WeatherSearch)


# In[18]:


weather_function


# In[19]:


class WeatherSearch1(BaseModel):
    airport_code: str = Field(description="airport code to get weather for")


# **Note**: The next cell is expected to generate an error.

# In[20]:


convert_pydantic_to_openai_function(WeatherSearch1)


# In[21]:


class WeatherSearch2(BaseModel):
    """Call this with an airport code to get the weather at that airport"""
    airport_code: str


# In[22]:


convert_pydantic_to_openai_function(WeatherSearch2)


# In[23]:


from langchain.chat_models import ChatOpenAI


# In[24]:


model = ChatOpenAI()


# In[25]:


model.invoke("what is the weather in SF today?", functions=[weather_function])


# In[26]:


model_with_function = model.bind(functions=[weather_function])


# In[27]:


model_with_function.invoke("what is the weather in sf?")


# ## Forcing it to use a function
# 
# We can force the model to use a function

# In[28]:


model_with_forced_function = model.bind(functions=[weather_function], function_call={"name":"WeatherSearch"})


# In[29]:


model_with_forced_function.invoke("what is the weather in sf?")


# In[30]:


model_with_forced_function.invoke("hi!")


# ## Using in a chain
# 
# We can use this model bound to function in a chain as we normally would

# In[31]:


from langchain.prompts import ChatPromptTemplate


# In[32]:


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])


# In[33]:


chain = prompt | model_with_function


# In[34]:


chain.invoke({"input": "what is the weather in sf?"})


# ## Using multiple functions
# 
# Even better, we can pass a set of function and let the LLM decide which to use based on the question context.

# In[35]:


class ArtistSearch(BaseModel):
    """Call this to get the names of songs by a particular artist"""
    artist_name: str = Field(description="name of artist to look up")
    n: int = Field(description="number of results")


# In[36]:


functions = [
    convert_pydantic_to_openai_function(WeatherSearch),
    convert_pydantic_to_openai_function(ArtistSearch),
]


# In[37]:


model_with_functions = model.bind(functions=functions)


# In[38]:


model_with_functions.invoke("what is the weather in sf?")


# In[39]:


model_with_functions.invoke("what are three songs by taylor swift?")


# In[ ]:


model_with_functions.invoke("hi!")


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




