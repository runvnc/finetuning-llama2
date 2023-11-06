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


def init():
	sess = sagemaker.Session()
	# sagemaker session bucket -> used for uploading data, models and logs
	# sagemaker will automatically create this bucket if it not exists
	sagemaker_session_bucket=None
	if sagemaker_session_bucket is None and sess is not None:
	    # set to default bucket if a bucket name is not given
	    sagemaker_session_bucket = sess.default_bucket()

	try:
	    role = sagemaker.get_execution_role()
	except ValueError:
	    iam = boto3.client('iam')
	    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

	sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

	print(f"sagemaker role arn: {role}")
	print(f"sagemaker bucket: {sess.default_bucket()}")
	print(f"sagemaker session region: {sess.boto_region_name}")

	from sagemaker.huggingface import get_huggingface_llm_image_uri

	# retrieve the llm image uri
	llm_image = get_huggingface_llm_image_uri(
	  "huggingface",
	  version="0.8.2"
	)

	# print ecr image uri
	print(f"llm image uri: {llm_image}")

	return (llm_image, role)


# model_data
#"s3://sagemaker-us-east-1-308819823671/huggingface-qlora-meta-llama-Llama-2-13-2023-09-01-16-39-25-384/output/model.tar.gz",
#	  env=config

# llama-2-13b-hf-nyc-finetuned


def deploy(model_data, endpoint_name, instance_type = "ml.g5.12xlarge",
           number_of_gpu = 4, health_check_timeout = 300):

	(llm_image, role) = init()
    
    TGI_config = {
      'HF_MODEL_ID': "/opt/ml/model", # path to where sagemaker stores the model
      'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
      'MAX_INPUT_LENGTH': json.dumps(1024), 
      'MAX_TOTAL_TOKENS': json.dumps(2048), 
      # 'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
    }
	
	llm_model = HuggingFaceModel(
	  role=role,
	  image_uri=llm_image,
	  #model_data="s3://sagemaker-us-east-1-308819823671/huggingface-qlora-llama2-13b-chat-2023--2023-08-02-08-54-16-604/output/model.tar.gz",
	  model_data="s3://sagemaker-us-east-1-308819823671/huggingface-qlora-meta-llama-Llama-2-13-2023-09-01-16-39-25-384/output/model.tar.gz",
	  env=TGI_config
	)

	llm = llm_model.deploy(
	  endpoint_name,
	  initial_instance_count=1,
	  instance_type=instance_type,
	  # volume_size=400, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
	  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
	)


