prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{basic_endpoint}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint)}')


# ## LLaMA2 finetuned on NYC summit blogs

# In[36]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{basic_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint_ft)}')


# ## LLaMA2-chat

# In[38]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{chat_endpoint}: {query_endpoint({"inputs": [[{"role": "user", "content": prompt}]], "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint)}')


# ## LLaMA2-chat finetuned on NYC summit blogs

# In[39]:


prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": json.dumps([[{"role": "user", "content": prompt}]]), "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint_ft)}')


# In[40]:


prompt = "<s> [INST] Hi! I'm Aris and I am wondering what I should do today in sunny Athens. [/INST]"
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint_ft)}')


# # Do the different models know what P5 instances are?
# ## LLaMA2-13b 

# In[27]:


prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"
print(f'{basic_endpoint}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint)}')


# ## LLaMA2-13b finetuned on NYC summit blogs

# In[29]:


prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"
print(f'{basic_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, basic_endpoint_ft)}')


# ## LLaMA2-13b-chat

# In[30]:


prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
print(f'{chat_endpoint}: {query_endpoint({"inputs": [[{"role": "user", "content": prompt}]], "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint)}')


# ## LLaMA2-13b-chat finetuned on NYC summit blogs

# In[31]:


prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": json.dumps([[{"role": "user", "content": prompt}]]), "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, chat_endpoint_ft)}')


# In[41]:


prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]"
print(f'{chat_endpoint_ft}: {query_endpoint({"inputs": prompt, "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01, "return_full_text": False}}, chat_endpoint_ft)}')



prompt = """
Blogpost conclusion: 
In conclusion, this blog post delves into the critical process of infusing domain-specific knowledge into large language models (LLMs) like LLaMA2, emphasizing the importance of addressing challenges related to helpfulness, honesty, and harmlessness when designing LLM-powered applications for enterprise-grade quality. The primary focus here is on the parametric approach to fine-tuning, which efficiently injects niche expertise into foundation models without compromising their general linguistic capabilities.The blog highlights the steps involved in fine-tuning LLaMA2 using parameter-efficient fine-tuning techniques, such as the qLoRA approach, and how this process can be conducted on Amazon SageMaker. By adopting this approach, practitioners can adapt LLaMA2 to specific domains, ensuring that the models remain up-to-date with recent knowledge even beyond their original training data. The article also underscores the versatility of this approach, showing that it can be applied to models like LLaMA2-chat, which have already undergone task-specific fine-tuning. This opens up opportunities to infuse knowledge into LLMs without the need for extensive instruction or chat-based fine-tuning, preserving their task-specific nature.
Task: 
Please extract the main takeaways from this blogpost.
"""

print(f'{chat_endpoint}: {query_endpoint({"inputs": [[{"role": "user", "content": prompt}]], "parameters": {"max_new_tokens": 200, "top_p": 0.9, "temperature": 0.01}}, chat_endpoint)}')

