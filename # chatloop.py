#!/usr/bin/env python

from termcolor import colored
from generate_text import generate_text

def chat_loop():
    inputs = []  # Initialize the conversation history list
    while True:  # Start an infinite loop
        try:
            user_input = input(colored('You: ', 'white'))  # Get input from the user
            if user_input.lower() == 'exit':  # Allow the user to exit the loop
                print("Exiting chat...")
                break

            # Append the user's input to the conversation history
            inputs.append({"role": "user", "content": user_input})

            # Generate the model's response
            model_response = generate_text('aws13bchatz2', user_input)

            # Append the model's response to the conversation history
            inputs.append({"role": "assistant", "content": model_response})

            # Print the model's response
            print(colored('AI: ', 'green') + colored(model_response, 'green'))

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting chat...")
            break

if __name__ == '__main__':
    chat_loop()
#!/usr/bin/env python

import json
from termcolor import colored
from generate_text import generate_text

def chat_loop():
    inputs = []  # Initialize the conversation history list
    while True:  # Start an infinite loop
        try:
            user_input = input('You: ')  # Get input from the user
            if user_input.lower() == 'exit':  # Allow the user to exit the loop
                print("Exiting chat...")
                break

            # Append the user's input to the conversation history
            inputs.append({"role": "user", "content": user_input})

            # Generate the model's response
            model_response = generate_text('aws13bchatz2', user_input)

            # Append the model's response to the conversation history
            inputs.append({"role": "assistant", "content": model_response})

            # Print the model's response
            print(colored('AI: ', 'green') + colored(model_response, 'green'))

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting chat...")
            break

if __name__ == '__main__':
    chat_loop()
