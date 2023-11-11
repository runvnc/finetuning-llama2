#!/usr/bin/env python

# Note: this doesn't work yet. will fix after sleep period.

from termcolor import colored
from generate_text import generate_text


def chat_loop():
    inputs = [{"role":"system", "content": "You are a helpful assistant."}]
    while True:
        try:
            user_input = input(colored('You: ', 'white'))
            if user_input.lower() == 'exit':
                print("Exiting chat...")
                break
            print()

            inputs.append({"role": "user", "content": user_input })

            model_response = generate_text('aws13bchatz2', inputs)

            inputs.append({"role": "assistant", "content": model_response})

            print(inputs)
            print(colored('AI: ', 'green') + colored(model_response, 'green'))
            print()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting chat...")
            break

if __name__ == '__main__':
    chat_loop()
