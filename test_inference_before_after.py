#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, inputs, output):
    prompt = inputs[0]['content']
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()

# jumpstart-dft-meta-textgeneration-llama-2-13b-f-1

prompt = "Please list some fun activities for a visit to San Diego. " 
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))


#prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?" 

#inputs = [[{"role": "user", "content": prompt}]]

#print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

prompt = "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with? " 
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))


#prompt = "What is Amazon Bedrock?"
#inputs = [[{"role": "user", "content": prompt}]]
#print_output('llama2-13b-chat', inputs, generate_text('llama2-13b-chat', inputs))

prompt = "What are agents for Amazon Bedrock? "
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))

prompt = "How can I create a generative AI agent using a foundational model on Amazon? "
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))

prompt = "What is AWS entity resolution? "
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))

prompt = "What kind of vector datastore system is available for Aurora? "
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))

prompt = "How does similarity search in Amazon OpenSearch Serverless work? "
inputs = [{"role": "user", "content": prompt}]
print_output('aws13bchatz2', inputs, generate_text('aws13bchatz2', inputs))


