#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, inputs, output):
    prompt = inputs[0][0]['content']
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()

prompt = "<s> [INST] Please list some fun activities for a visit to San Diego. [/INST]" 
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))



#prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?" 

#inputs = [[{"role": "user", "content": prompt}]]

#print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

prompt = "<s> [INST] What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? [/INST]" 
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))


#prompt = "What is Amazon Bedrock?"
#inputs = [[{"role": "user", "content": prompt}]]
#print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

prompt = "<s> [INST] What are agents for Amazon Bedrock? [/INST]"
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))

prompt = "<s> [INST] How can I create a generative AI agent using a foundational model on Amazon? [/INST]"
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))

prompt = "<s> [INST] What is AWS entity resolution? [/INST]"
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))

prompt = "<s> [INST] What kind of vector datastore system is available for Aurora? [/INST]"
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))

prompt = "<s> [INST] How does similarity search in Amazon OpenSearch Serverless work? [/INST]"
inputs = [[{"role": "user", "content": prompt}]]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', prompt))



