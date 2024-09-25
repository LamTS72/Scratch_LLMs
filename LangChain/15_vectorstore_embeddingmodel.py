#!/usr/bin/env python
# coding: utf-8

# # Vectorstores and Embeddings
# 
# Recall the overall workflow for retrieval augmented generation (RAG):

# ![overview.jpeg](attachment:overview.jpeg)

# In[1]:


import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# We just discussed `Document Loading` and `Splitting`.

# In[2]:


from langchain.document_loaders import PyPDFLoader

# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# In[3]:


# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)


# In[4]:


splits = text_splitter.split_documents(docs)


# In[5]:


len(splits)


# ## Embeddings
# 
# Let's take our splits and embed them.

# In[6]:


from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()


# In[7]:


sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"


# In[8]:


embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)


# In[9]:


import numpy as np


# In[10]:


np.dot(embedding1, embedding2)


# In[11]:


np.dot(embedding1, embedding3)


# In[12]:


np.dot(embedding2, embedding3)


# ## Vectorstores

# In[13]:


# ! pip install chromadb


# In[14]:


from langchain.vectorstores import Chroma


# In[15]:


persist_directory = 'docs/chroma/'


# In[16]:


get_ipython().system('rm -rf ./docs/chroma  # remove old database files if any')


# In[17]:


vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


# In[18]:


print(vectordb._collection.count())


# ### Similarity Search

# In[19]:


question = "is there an email i can ask for help"


# In[20]:


docs = vectordb.similarity_search(question,k=3)


# In[21]:


len(docs)


# In[22]:


docs[0].page_content


# Let's save this so we can use it later!

# In[23]:


vectordb.persist()


# ## Failure modes
# 
# This seems great, and basic similarity search will get you 80% of the way there very easily. 
# 
# But there are some failure modes that can creep up. 
# 
# Here are some edge cases that can arise - we'll fix them in the next class.

# In[24]:


question = "what did they say about matlab?"


# In[25]:


docs = vectordb.similarity_search(question,k=5)


# Notice that we're getting duplicate chunks (because of the duplicate `MachineLearning-Lecture01.pdf` in the index).
# 
# Semantic search fetches all similar documents, but does not enforce diversity.
# 
# `docs[0]` and `docs[1]` are indentical.

# In[26]:


docs[0]


# In[27]:


docs[1]


# We can see a new failure mode.
# 
# The question below asks a question about the third lecture, but includes results from other lectures as well.

# In[28]:


question = "what did they say about regression in the third lecture?"


# In[29]:


docs = vectordb.similarity_search(question,k=5)


# In[30]:


for doc in docs:
    print(doc.metadata)


# In[31]:


print(docs[4].page_content)


# Approaches discussed in the next lecture can be used to address both!

# In[ ]:




