#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, prompt, output):
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()

prompt = "Hi! I'm Jason and I am wondering what I should do today in sunny Athens."
print_output('llama2-7b', prompt, generate_text('llama2-7b', prompt))

print_output('awsarticles5', prompt, generate_text('awsarticles5', prompt))


prompt = "Amazon EC2 P5 instances are equipped with GPUs of the type"
print_output('llama2-7b', prompt, generate_text('llama2-7b', prompt))


## LLaMA2-13b-chat
#prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
# ## LLaMA2-13b-chat finetuned on NYC summit blogs
#prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]"


