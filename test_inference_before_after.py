#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, inputs, output):
    prompt = inputs[0][0]['content']
    print(output)
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()


inputs = [[{"role": "user", "content": "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"}]]

print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

#print_output('awsarticles24', prompt, generate_text('awsarticles24', prompt))


#prompt = "Agents for Amazon Bedrock automate the"

#print_output('llama2-13b-chat', prompt, generate_text('llama2-13b-chat', prompt))

#print_output('awsarticles24', prompt, generate_text('awsarticles24', prompt))


## LLaMA2-13b-chat
#prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?"
# ## LLaMA2-13b-chat finetuned on NYC summit blogs
#prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]"


