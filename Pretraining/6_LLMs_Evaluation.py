#!/usr/bin/env python
# coding: utf-8

# # Lesson 6. Model evaluation
# 
# The model comparison tool that Sung described in the video can be found at this link: https://console.upstage.ai/ (note that you need to create a free account to try it out.)
# 
# A useful tool for evaluating LLMs is the **LM Evaluation Harness** built by EleutherAI. Information about the harness can be found at this [github repo](https://github.com/EleutherAI/lm-evaluation-harness):
# 
# You can run the commented code below to install the evaluation harness in your own environment:

# In[2]:


#!pip install -U git+https://github.com/EleutherAI/lm-evaluation-harness


# You will evaluate TinySolar-248m-4k on 5 questions from the **TruthfulQA MC2 task**. This is a multiple-choice question answering task that tests the model's ability to identify true statements. You can read more about the TruthfulQA benchmark in [this paper](https://arxiv.org/abs/2109.07958), and you can checkout the code for implementing the tasks at this [github repo](https://github.com/sylinrl/TruthfulQA).
# 
# The code below runs only the TruthfulQA MC2 task using the LM Evaluation Harness:

# In[3]:


get_ipython().system('lm_eval --model hf      --model_args pretrained=./models/TinySolar-248m-4k      --tasks truthfulqa_mc2      --device cpu      --limit 5')


# ### Evaluation for the Hugging Face Leaderboard
# You can use the code below to test your own model against the evaluations required for the [Hugging Face leaderboard](https://huggingface.co/open-llm-leaderboard). 
# 
# If you decide to run this evaluation on your own model, don't change the few-shot numbers below - they are set by the rules of the leaderboard.

# In[ ]:


import os

def h6_open_llm_leaderboard(model_name):
  task_and_shot = [
      ('arc_challenge', 25),
      ('hellaswag', 10),
      ('mmlu', 5),
      ('truthfulqa_mc2', 0),
      ('winogrande', 5),
      ('gsm8k', 5)
  ]

  for task, fewshot in task_and_shot:
    eval_cmd = f"""
    lm_eval --model hf \
        --model_args pretrained={model_name} \
        --tasks {task} \
        --device cpu \
        --num_fewshot {fewshot}
    """
    os.system(eval_cmd)

h6_open_llm_leaderboard(model_name="upstage/TinySolar-248m-4k")

