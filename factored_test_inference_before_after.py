#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, inputs, output):
    prompt = inputs[0]['content']
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()

def query_model_and_print(model_name, prompt):
    inputs = [{"role": "user", "content": prompt}]
    output = generate_text(model_name, inputs)
    print_output(model_name, inputs, output)

# Use the new function to handle the prompting logic for different prompts and models
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "Please list some fun activities for a visit to San Diego.")
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "What are Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?")
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "What are agents for Amazon Bedrock?")
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "How can I create a generative AI agent using a foundational model on Amazon?")
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "What is AWS entity resolution?")
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "What kind of vector datastore system is available for Aurora?")
query_model_and_print('jumpstart-dft-meta-textgeneration-llama-2-13b-f-1', "How does similarity search in Amazon OpenSearch Serverless work?")
