import sagemaker
import boto3

#basic_endpoint_ft = 'llama-2-13b-hf-nyc-finetuned'

endpoints = {
    'llama2-13b': 'jumpstart-dft-meta-textgeneration-llama-2-13b',
    'llama2-13b-chat': 'jumpstart-dft-meta-textgeneration-llama-2-13b-f'
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


def generate_text(model, prompt):
    query_endpoint( {"inputs": prompt,
        "parameters": {
            "max_new_tokens": 200, "top_p": 0.9, 
            "temperature": 0.01, 
            "return_full_text": False}
        }, 
        endpoints[model])
