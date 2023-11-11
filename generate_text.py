import sagemaker
import boto3
import json
from llama2prompt import *

endpoints = {
    'llama2-13b-chat': 'jumpstart-dft-meta-textgeneration-llama-2-13b-f-1',
}


def query_endpoint(payload, endpoint_name):
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response


def generate_chat_response(model, inputs, msg_format=True):
    if msg_format:
        inputs = llama_v2_prompt(inputs)
    else:
        inputs = [inputs]
    if model in endpoints:
        endpoint = endpoints[model]
    else:
        endpoint = model
    result = query_endpoint( {"inputs": inputs,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.001,
            "return_full_text": False }
        }, 
        endpoint)
    if 'generation' in result[0]:
        # 0.8.2
        return result[0]['generation']['content']
    else:
        # 0.9.3
        return result[0]['generated_text']

