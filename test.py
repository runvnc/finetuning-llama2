from generate_text import *

prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
print(generate_text('llama2-7b', prompt))

# ## LLaMA2 finetuned on NYC summit blogs
#print(generate_text(prompt, 'llaba2b-13b-tuned'))

# ## LLaMA2-chat
#print(generate_text(prompt, 'llama2-13b-chat'))


# ## LLaMA2-chat finetuned on NYC summit blogs

#prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
#prompt = "<s> [INST] Hi! I'm Aris and I am wondering what I should do today in sunny Athens. [/INST]"


# ## LLaMA2-13b 
#prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"


# ## LLaMA2-13b finetuned on NYC summit blogs



# ## LLaMA2-13b-chat
#prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"


# ## LLaMA2-13b-chat finetuned on NYC summit blogs
#prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]"


#prompt = """
#Blogpost conclusion: 
#In conclusion, this blog post delves into the critical process of infusing domain-specific knowledge into large language models (LLMs) like LLaMA2, emphasizing the importance of addressing challenges related to helpfulness, honesty, and harmlessness when designing LLM-powered applications for enterprise-grade quality. The primary focus here is on the parametric approach to fine-tuning, which efficiently injects niche expertise into foundation models without compromising their general linguistic capabilities.The blog highlights the steps involved in fine-tuning LLaMA2 using parameter-efficient fine-tuning techniques, such as the qLoRA approach, and how this process can be conducted on Amazon SageMaker. By adopting this approach, practitioners can adapt LLaMA2 to specific domains, ensuring that the models remain up-to-date with recent knowledge even beyond their original training data. The article also underscores the versatility of this approach, showing that it can be applied to models like LLaMA2-chat, which have already undergone task-specific fine-tuning. This opens up opportunities to infuse knowledge into LLMs without the need for extensive instruction or chat-based fine-tuning, preserving their task-specific nature.
#Task: 
#Please extract the main takeaways from this blogpost.
#"""


