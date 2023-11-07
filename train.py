import sagemaker
import boto3
import time
import json

from datasets import Dataset
from langchain.document_loaders import WebBaseLoader
from random import randint
from itertools import chain
from functools import partial
from transformers import AutoTokenizer
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from huggingface_hub import HfFolder

from init_sagemaker import *

def strip_spaces(doc):
    return {"text": doc.page_content.replace("  ", "")}

def chunk(sample, chunk_length=2048):
    # define global remainder variable to save remainder from batches to use in next batch
    global remainder
    # Concatenate all texts and add remainder from previous batch
    concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
    concatenated_examples = {k: remainder[k] + concatenated_examples[k] for k in concatenated_examples.keys()}
    # get total number of tokens for batch
    batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

    # get max number of chunks for batch
    if batch_total_length >= chunk_length:
        batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_length] for i in range(0, batch_chunk_length, chunk_length)]
        for k, t in concatenated_examples.items()
    }
    # add remainder to global variable for next batch
    remainder = {k: concatenated_examples[k][batch_chunk_length:] for k in concatenated_examples.keys()}
    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result

def sum_dataset_arrays(dataset):
    total = 0
    for i in range(0,8):
        total = total + len(lm_dataset[i]['input_ids'])
    return total

def load_from_web(pretrained_modelid = "meta-llama/Llama-2-7b-hf", urls)
    loader = WebBaseLoader(urls)

    data = loader.load()

    stripped_data = list(map(strip_spaces, data))

    dataset = Dataset.from_list(stripped_data)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # empty list to save remainder from batches to use in next batch
    remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}

    lm_dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]), batched=True, remove_columns=list(dataset.features)
    ).map(
        partial(chunk, chunk_length=4096),
        batched=True,
    )

    print(f"Total number of training samples: {len(lm_dataset)}")
    print(f"Total number of training tokens: {sum_dataset_arrays(lm_dataset)}")
    return lm_dataset

#f's3://{sess.default_bucket()}/processed/llama/genai-nyc-summit/train'
def store_dataset(s3_bucket_path):
    # save train_dataset to s3
    training_input_path = sess.default_bucket() + '/' + s3_buck_path
    lm_dataset.save_to_disk(training_input_path)

    print("uploaded data to:")
    print(f"training dataset to: {training_input_path}")


ignore = """
# # Fine-tuning

# define Training Job Name
job_name = f'huggingface-qlora-{model_id.replace("/", "-")}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

# hyperparameters, which are passed into the training job
hyperparameters ={
  'model_id': model_id,                             # pre-trained model
  'dataset_path': '/opt/ml/input/data/training',    # path where sagemaker will save training dataset
  'epochs': 20,                                      # number of training epochs
  'per_device_train_batch_size': 2,                 # batch size for training
  'lr': 2e-4,                                       # learning rate used during training
  'hf_token': HfFolder.get_token(),                 # huggingface token to access llama 2
  'merge_weights': True,                            # wether to merge LoRA into the model (needs more memory)
}

# create the Estimator
huggingface_estimator = HuggingFace(
    entry_point          = 'run_clm.py',      # train script
    source_dir           = 'scripts',         # directory which includes all the files needed for training
    instance_type        = 'ml.g5.4xlarge',   # instances type used for the training job
    instance_count       = 1,                 # the number of instances used for training
    base_job_name        = job_name,          # the name of the training job
    role                 = role,              # Iam role used in training job to access AWS ressources, e.g. S3
    volume_size          = 300,               # the size of the EBS volume in GB
    transformers_version = '4.28',            # the transformers version used in the training job
    pytorch_version      = '2.0',             # the pytorch_version version used in the training job
    py_version           = 'py310',           # the python version used in the training job
    hyperparameters      =  hyperparameters,  # the hyperparameters passed to the training job
    environment          = { "HUGGINGFACE_HUB_CACHE": "/tmp/.cache" }, # set env variable to cache models in /tmp
)



# define a data input dictonary with our uploaded s3 uris
data = {'training': training_input_path}

# starting the train job with our uploaded datasets as input
huggingface_estimator.fit(data, wait=False)



from sagemaker.huggingface import get_huggingface_llm_image_uri

# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface",
  version="0.8.2"
)

# print ecr image uri
print(f"llm image uri: {llm_image}")



# sagemaker config
instance_type = "ml.g5.12xlarge"
number_of_gpu = 4
health_check_timeout = 300

# TGI config
config = {
  'HF_MODEL_ID': "/opt/ml/model", # path to where sagemaker stores the model
  'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
  'MAX_INPUT_LENGTH': json.dumps(1024),  # Max length of input text
  'MAX_TOTAL_TOKENS': json.dumps(2048),  # Max length of the generation (including input text),
  # 'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
}

# create HuggingFaceModel
llm_model = HuggingFaceModel(
  role=role,
  image_uri=llm_image,
  #model_data="s3://sagemaker-us-east-1-308819823671/huggingface-qlora-llama2-13b-chat-2023--2023-08-02-08-54-16-604/output/model.tar.gz",
  model_data="s3://sagemaker-us-east-1-308819823671/huggingface-qlora-meta-llama-Llama-2-13-2023-09-01-16-39-25-384/output/model.tar.gz",
  env=config
)


# Deploy model to an endpoint
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
llm = llm_model.deploy(
  #endpoint_name="llama-2-13b-chat-hf-nyc-finetuned", # alternatively "llama-2-13b-hf-nyc-finetuned" 
  endpoint_name="llama-2-13b-hf-nyc-finetuned", # alternatively "llama-2-13b-hf-nyc-finetuned"  
  initial_instance_count=1,
  instance_type=instance_type,
  # volume_size=400, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
)
"""


