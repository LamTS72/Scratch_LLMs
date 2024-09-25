#!/usr/bin/env python
# coding: utf-8

# # LangChain Expression Language (LCEL)

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


#!pip install pydantic==1.10.8


# In[3]:


from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


# ## Simple Chain

# In[4]:


prompt = ChatPromptTemplate.from_template(
    "tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()


# In[5]:


chain = prompt | model | output_parser


# In[6]:


chain.invoke({"topic": "bears"})


# ## More complex chain
# 
# And Runnable Map to supply user-provided inputs to the prompt.

# In[7]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch


# In[8]:


vectorstore = DocArrayInMemorySearch.from_texts(
    ["harrison worked at kensho", "bears like to eat honey"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


# In[9]:


retriever.get_relevant_documents("where did harrison work?")


# In[10]:


retriever.get_relevant_documents("what do bears like to eat")


# In[11]:


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# In[12]:


from langchain.schema.runnable import RunnableMap


# In[13]:


chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser


# In[14]:


chain.invoke({"question": "where did harrison work?"})


# In[15]:


inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})


# In[16]:


inputs.invoke({"question": "where did harrison work?"})


# ## Bind
# 
# and OpenAI Functions

# In[17]:


functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    }
  ]


# In[18]:


prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}")
    ]
)
model = ChatOpenAI(temperature=0).bind(functions=functions)


# In[19]:


runnable = prompt | model


# In[20]:


runnable.invoke({"input": "what is the weather in sf"})


# In[21]:


functions = [
    {
      "name": "weather_search",
      "description": "Search for weather given an airport code",
      "parameters": {
        "type": "object",
        "properties": {
          "airport_code": {
            "type": "string",
            "description": "The airport code to get the weather for"
          },
        },
        "required": ["airport_code"]
      }
    },
        {
      "name": "sports_search",
      "description": "Search for news of recent sport events",
      "parameters": {
        "type": "object",
        "properties": {
          "team_name": {
            "type": "string",
            "description": "The sports team to search for"
          },
        },
        "required": ["team_name"]
      }
    }
  ]


# In[22]:


model = model.bind(functions=functions)


# In[23]:


runnable = prompt | model


# In[24]:


runnable.invoke({"input": "how did the patriots do yesterday?"})


# ## Fallbacks

# In[25]:


from langchain.llms import OpenAI
import json


# **Note**: Due to the deprecation of OpenAI's model `text-davinci-001` on 4 January 2024, you'll be using OpenAI's recommended replacement model `gpt-3.5-turbo-instruct` instead.

# In[26]:


simple_model = OpenAI(
    temperature=0, 
    max_tokens=1000, 
    model="gpt-3.5-turbo-instruct"
)
simple_chain = simple_model | json.loads


# In[27]:


challenge = "write three poems in a json blob, where each poem is a json blob of a title, author, and first line"


# In[28]:


simple_model.invoke(challenge)


# **Note**: The next line is expected to fail.

# In[29]:


simple_chain.invoke(challenge)


# In[31]:


model = ChatOpenAI(temperature=0)
chain = model | StrOutputParser() | json.loads


# In[32]:


chain.invoke(challenge)


# In[33]:


final_chain = simple_chain.with_fallbacks([chain])


# In[34]:


final_chain.invoke(challenge)


# ## Interface

# In[35]:


prompt = ChatPromptTemplate.from_template(
    "Tell me a short joke about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser


# In[36]:


chain.invoke({"topic": "bears"})


# In[37]:


chain.batch([{"topic": "bears"}, {"topic": "frogs"}])


# In[38]:


for t in chain.stream({"topic": "bears"}):
    print(t)


# In[39]:


response = await chain.ainvoke({"topic": "bears"})
response


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




