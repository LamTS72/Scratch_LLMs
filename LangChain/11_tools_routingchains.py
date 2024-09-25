#!/usr/bin/env python
# coding: utf-8

# # Tools and Routing

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


from langchain.agents import tool


# In[3]:


@tool
def search(query: str) -> str:
    """Search for weather online"""
    return "42f"


# In[4]:


search.name


# In[5]:


search.description


# In[6]:


search.args


# In[7]:


from pydantic import BaseModel, Field
class SearchInput(BaseModel):
    query: str = Field(description="Thing to search for")


# In[8]:


@tool(args_schema=SearchInput)
def search(query: str) -> str:
    """Search for the weather online."""
    return "42f"


# In[9]:


search.args


# In[10]:


search.run("sf")


# In[11]:


import requests
from pydantic import BaseModel, Field
import datetime

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'


# In[12]:


get_current_temperature.name


# In[13]:


get_current_temperature.description


# In[14]:


get_current_temperature.args


# In[15]:


from langchain.tools.render import format_tool_to_openai_function


# In[16]:


format_tool_to_openai_function(get_current_temperature)


# In[17]:


get_current_temperature({"latitude": 13, "longitude": 14})


# In[18]:


import wikipedia
@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)


# In[19]:


search_wikipedia.name


# In[20]:


search_wikipedia.description


# In[21]:


format_tool_to_openai_function(search_wikipedia)


# In[22]:


search_wikipedia({"query": "langchain"})


# In[23]:


from langchain.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain.utilities.openapi import OpenAPISpec


# In[24]:


text = """
{
  "openapi": "3.0.0",
  "info": {
    "version": "1.0.0",
    "title": "Swagger Petstore",
    "license": {
      "name": "MIT"
    }
  },
  "servers": [
    {
      "url": "http://petstore.swagger.io/v1"
    }
  ],
  "paths": {
    "/pets": {
      "get": {
        "summary": "List all pets",
        "operationId": "listPets",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "description": "How many items to return at one time (max 100)",
            "required": false,
            "schema": {
              "type": "integer",
              "maximum": 100,
              "format": "int32"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "A paged array of pets",
            "headers": {
              "x-next": {
                "description": "A link to the next page of responses",
                "schema": {
                  "type": "string"
                }
              }
            },
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pets"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create a pet",
        "operationId": "createPets",
        "tags": [
          "pets"
        ],
        "responses": {
          "201": {
            "description": "Null response"
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    },
    "/pets/{petId}": {
      "get": {
        "summary": "Info for a specific pet",
        "operationId": "showPetById",
        "tags": [
          "pets"
        ],
        "parameters": [
          {
            "name": "petId",
            "in": "path",
            "required": true,
            "description": "The id of the pet to retrieve",
            "schema": {
              "type": "string"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Expected response to a valid request",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Pet"
                }
              }
            }
          },
          "default": {
            "description": "unexpected error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Error"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "Pet": {
        "type": "object",
        "required": [
          "id",
          "name"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "format": "int64"
          },
          "name": {
            "type": "string"
          },
          "tag": {
            "type": "string"
          }
        }
      },
      "Pets": {
        "type": "array",
        "maxItems": 100,
        "items": {
          "$ref": "#/components/schemas/Pet"
        }
      },
      "Error": {
        "type": "object",
        "required": [
          "code",
          "message"
        ],
        "properties": {
          "code": {
            "type": "integer",
            "format": "int32"
          },
          "message": {
            "type": "string"
          }
        }
      }
    }
  }
}
"""


# In[25]:


spec = OpenAPISpec.from_text(text)


# In[26]:


pet_openai_functions, pet_callables = openapi_spec_to_openai_fn(spec)


# In[27]:


pet_openai_functions


# In[28]:


from langchain.chat_models import ChatOpenAI


# In[29]:


model = ChatOpenAI(temperature=0).bind(functions=pet_openai_functions)


# In[30]:


model.invoke("what are three pets names")


# In[31]:


model.invoke("tell me about pet with id 42")


# ### Routing
# 
# In lesson 3, we show an example of function calling deciding between two candidate functions.
# 
# Given our tools above, let's format these as OpenAI functions and show this same behavior.

# In[32]:


functions = [
    format_tool_to_openai_function(f) for f in [
        search_wikipedia, get_current_temperature
    ]
]
model = ChatOpenAI(temperature=0).bind(functions=functions)


# In[33]:


model.invoke("what is the weather in sf right now")


# In[34]:


model.invoke("what is langchain")


# In[35]:


from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    ("user", "{input}"),
])
chain = prompt | model


# In[36]:


chain.invoke({"input": "what is the weather in sf right now"})


# In[37]:


from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser


# In[38]:


chain = prompt | model | OpenAIFunctionsAgentOutputParser()


# In[39]:


result = chain.invoke({"input": "what is the weather in sf right now"})


# In[40]:


type(result)


# In[41]:


result.tool


# In[42]:


result.tool_input


# In[43]:


get_current_temperature(result.tool_input)


# In[44]:


result = chain.invoke({"input": "hi!"})


# In[45]:


type(result)


# In[46]:


result.return_values


# In[47]:


from langchain.schema.agent import AgentFinish
def route(result):
    if isinstance(result, AgentFinish):
        return result.return_values['output']
    else:
        tools = {
            "search_wikipedia": search_wikipedia, 
            "get_current_temperature": get_current_temperature,
        }
        return tools[result.tool].run(result.tool_input)


# In[48]:


chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route


# In[49]:


result = chain.invoke({"input": "What is the weather in san francisco right now?"})


# In[50]:


result


# In[51]:


result = chain.invoke({"input": "What is langchain?"})


# In[52]:


result


# In[53]:


chain.invoke({"input": "hi!"})


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




