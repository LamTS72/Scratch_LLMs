#!/usr/bin/env python
# coding: utf-8

# # Tagging and Extraction Using OpenAI functions

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


from typing import List
from pydantic import BaseModel, Field
from langchain.utils.openai_functions import convert_pydantic_to_openai_function


# In[3]:


class Tagging(BaseModel):
    """Tag the piece of text with particular info."""
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")


# In[4]:


convert_pydantic_to_openai_function(Tagging)


# In[5]:


from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI


# In[6]:


model = ChatOpenAI(temperature=0)


# In[7]:


tagging_functions = [convert_pydantic_to_openai_function(Tagging)]


# In[8]:


prompt = ChatPromptTemplate.from_messages([
    ("system", "Think carefully, and then tag the text as instructed"),
    ("user", "{input}")
])


# In[9]:


model_with_functions = model.bind(
    functions=tagging_functions,
    function_call={"name": "Tagging"}
)


# In[10]:


tagging_chain = prompt | model_with_functions


# In[11]:


tagging_chain.invoke({"input": "I love langchain"})


# In[12]:


tagging_chain.invoke({"input": "non mi piace questo cibo"})


# In[13]:


from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser


# In[14]:


tagging_chain = prompt | model_with_functions | JsonOutputFunctionsParser()


# In[15]:


tagging_chain.invoke({"input": "non mi piace questo cibo"})


# ## Extraction
# 
# Extraction is similar to tagging, but used for extracting multiple pieces of information.

# In[16]:


from typing import Optional
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")


# In[17]:


class Information(BaseModel):
    """Information to extract."""
    people: List[Person] = Field(description="List of info about people")


# In[18]:


convert_pydantic_to_openai_function(Information)


# In[19]:


extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})


# In[20]:


extraction_model.invoke("Joe is 30, his mom is Martha")


# In[21]:


prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"),
    ("human", "{input}")
])


# In[22]:


extraction_chain = prompt | extraction_model


# In[23]:


extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})


# In[24]:


extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()


# In[25]:


extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})


# In[26]:


from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser


# In[27]:


extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="people")


# In[28]:


extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})


# ## Doing it for real
# 
# We can apply tagging to a larger body of text.
# 
# For example, let's load this blog post and extract tag information from a sub-set of the text.

# In[29]:


from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
documents = loader.load()


# In[30]:


doc = documents[0]


# In[31]:


page_content = doc.page_content[:10000]


# In[32]:


print(page_content[:1000])


# In[33]:


class Overview(BaseModel):
    """Overview of a section of text."""
    summary: str = Field(description="Provide a concise summary of the content.")
    language: str = Field(description="Provide the language that the content is written in.")
    keywords: str = Field(description="Provide keywords related to the content.")


# In[34]:


overview_tagging_function = [
    convert_pydantic_to_openai_function(Overview)
]
tagging_model = model.bind(
    functions=overview_tagging_function,
    function_call={"name":"Overview"}
)
tagging_chain = prompt | tagging_model | JsonOutputFunctionsParser()


# In[35]:


tagging_chain.invoke({"input": page_content})


# In[36]:


class Paper(BaseModel):
    """Information about papers mentioned."""
    title: str
    author: Optional[str]


class Info(BaseModel):
    """Information to extract"""
    papers: List[Paper]


# In[37]:


paper_extraction_function = [
    convert_pydantic_to_openai_function(Info)
]
extraction_model = model.bind(
    functions=paper_extraction_function, 
    function_call={"name":"Info"}
)
extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")


# In[38]:


extraction_chain.invoke({"input": page_content})


# In[39]:


template = """A article will be passed to you. Extract from it all papers that are mentioned by this article. 

Do not extract the name of the article itself. If no papers are mentioned that's fine - you don't need to extract any! Just return an empty list.

Do not make up or guess ANY extra information. Only extract what exactly is in the text."""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", "{input}")
])


# In[40]:


extraction_chain = prompt | extraction_model | JsonKeyOutputFunctionsParser(key_name="papers")


# In[44]:


extraction_chain.invoke({"input": page_content})


# In[45]:


extraction_chain.invoke({"input": "hi"})


# In[48]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_overlap=0)


# In[49]:


splits = text_splitter.split_text(doc.page_content)


# In[50]:


len(splits)


# In[51]:


def flatten(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list


# In[52]:


flatten([[1, 2], [3, 4]])


# In[53]:


print(splits[0])


# In[54]:


from langchain.schema.runnable import RunnableLambda


# In[55]:


prep = RunnableLambda(
    lambda x: [{"input": doc} for doc in text_splitter.split_text(x)]
)


# In[56]:


prep.invoke("hi")


# In[57]:


chain = prep | extraction_chain.map() | flatten


# In[58]:


chain.invoke(doc.page_content)


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




