#!/usr/bin/env python
# coding: utf-8

# # Compare finetuned vs. non-finetuned models

# In[1]:


import os
import lamini

lamini.api_url = os.getenv("POWERML__PRODUCTION__URL")
lamini.api_key = os.getenv("POWERML__PRODUCTION__KEY")


# In[2]:


from llama import BasicModelRunner


# ### Try Non-Finetuned models

# In[3]:


non_finetuned = BasicModelRunner("meta-llama/Llama-2-7b-hf")


# In[4]:


non_finetuned_output = non_finetuned("Tell me how to train my dog to sit")


# In[5]:


print(non_finetuned_output)


# In[6]:


print(non_finetuned("What do you think of Mars?"))


# In[7]:


print(non_finetuned("taylor swift's best friend"))


# In[8]:


print(non_finetuned("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))


# ### Compare to finetuned models 

# In[9]:


finetuned_model = BasicModelRunner("meta-llama/Llama-2-7b-chat-hf")


# In[10]:


finetuned_output = finetuned_model("Tell me how to train my dog to sit")


# In[11]:


print(finetuned_output)


# In[12]:


print(finetuned_model("[INST]Tell me how to train my dog to sit[/INST]"))


# In[13]:


print(non_finetuned("[INST]Tell me how to train my dog to sit[/INST]"))


# In[14]:


print(finetuned_model("What do you think of Mars?"))


# In[15]:


print(finetuned_model("taylor swift's best friend"))


# In[16]:


print(finetuned_model("""Agent: I'm here to help you with your Amazon deliver order.
Customer: I didn't get my item
Agent: I'm sorry to hear that. Which item was it?
Customer: the blanket
Agent:"""))


# ### Compare to ChatGPT

# > Note: This section of the notebook has been updated.
# 
# When run with the query:
# 
# ```Tell me how to train my dog to sit```
# 
# ChatGPT responds with:
# 
# > Training your dog to sit is a basic and essential command that can be taught using positive reinforcement. Here's a simple step-by-step guide:
# > 1. **Prepare Treats:**
#    Gather small, soft treats that your dog enjoys. Make sure they are easy to chew and won't take too long to eat.
# > 2. **Find a Quiet Space:**
#    Choose a quiet area with minimal distractions for the training session. This will help your dog focus better.
# > 3. **Get Your Dog's Attention:**
#    Call your dog's name to get their attention. Make sure they are looking at you.
# > 4. **Use a Treat to Lure:**
#    Hold a treat close to your dog's nose, and slowly move your hand upward and slightly backward over their head. As you do this, your dog's natural response will be to follow the treat with their nose, causing them to sit.
# > 5. **Say the Command:**
#    As your dog starts to sit, say the command "Sit" in a clear and firm voice. Use the word consistently every time you want your dog to sit.
# > 6. **Reward and Praise:**
#    As soon as your dog sits, immediately reward them with the treat and offer verbal praise. This positive reinforcement will help them associate sitting with positive outcomes.
# > 7. **Repeat and Practice:**
#    Repeat the process several times in a row during each training session. Keep the sessions short (around 5-10 minutes) to prevent your dog from losing interest.
# > 8. **Add Duration:**
#    Once your dog consistently sits on command, you can gradually increase the duration by waiting a couple of seconds before giving the treat. This helps reinforce the sit command.
# > 9. **Generalize the Command:**
#    Practice the "sit" command in different locations and with various distractions to help your dog generalize the behavior.
# > 10. **Be Patient and Consistent:**
#     Patience and consistency are key in dog training. Always use positive reinforcement, and avoid punishment. If your dog doesn't succeed initially, go back a step and try again.
# > 
# > Remember that each dog is unique, and some may learn more quickly than others. Adjust your training approach based on your dog's individual needs and progress.

# In[ ]:




