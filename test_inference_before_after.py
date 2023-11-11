#!/usr/bin/env python

from termcolor import colored
from generate_text import *

def print_output(model_name, inputs, output):
    prompt = inputs[0]['content']
    print(colored('Model: ', 'blue') + colored(model_name, 'green'))
    print(colored('Prompt: ', 'blue') + colored(prompt, 'yellow'))
    print(colored('Output: ', 'blue') + colored(output, 'cyan'))
    print()

def query_model_and_print(model_name, prompt, msg_format=True):
    inputs = [{"role": "user", "content": prompt}]
    output = generate_chat_response(model_name, inputs, msg_format)
    print_output(model_name, inputs, output)

def compare_model_outputs(prompt):
    query_model_and_print('llama2-13b-chat', prompt, False)
    print()

    query_model_and_print('aws13bchatz2', prompt)
    print()

questions = [
    "Please list some fun activities for a visit to San Diego.",
    "Amazon EC2 P5 instances? Which kind of GPUs are they equipped with?",
    "What are agents for Amazon Bedrock?",
    "How can I create a generative AI agent using a foundational model on Amazon?",
    "What is AWS entity resolution?",
    "What kind of vector datastore system is available for Aurora?",
    "How does similarity search in Amazon OpenSearch Serverless work?"
]

for question in questions:
    compare_model_outputs(question)

