#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, inputs, output):
    prompt = inputs[0][0]['content']
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()

prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?" 

inputs = [[{"role": "user", "content": prompt}]]

print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]" 
print_output('aws13bchat3', inputs, generate_text('aws13bchat3', prompt))


prompt = "What is Amazon Bedrock?"
inputs = [[{"role": "user", "content": prompt}]]
print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

prompt = "<s> [INST] What is Amazon Bedrock? [/INST]"
print_output('aws13bchat3', inputs, generate_text('aws13bchat3', prompt))

prompt = "<s> [INST] How can I create a generative AI agent using a foundational model on Amazon? [/INST]"
print_output('aws13bchat3', inputs, generate_text('aws13bchat3', prompt))

