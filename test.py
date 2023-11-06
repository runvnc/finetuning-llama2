from generate_text import *

prompt = "Hi! I'm Jason and I am wondering what I should do today in sunny Athens."
print(generate_text('llama2-7b', prompt))

#prompt = "Hi! I'm Aris and I am wondering what I should do today in sunny Athens."
#prompt = "<s> [INST] Hi! I'm Aris and I am wondering what I should do today in sunny Athens. [/INST]"


# ## LLaMA2-13b 
prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"
print(generate_text('llama2-7b', prompt))


## LLaMA2-13b-chat
#prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
# ## LLaMA2-13b-chat finetuned on NYC summit blogs
#prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]"


