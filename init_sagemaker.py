import sagemaker
import boto3
import time
import json
import botocore

from datasets import Dataset
from langchain.document_loaders import WebBaseLoader
from random import randint
from itertools import chain
from functools import partial
from transformers import AutoTokenizer
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from huggingface_hub import HfFolder

#SAGEMAKER_ROLE = 'AmazonSageMaker-ExecutionRole-20231110T150746'
SAGEMAKER_ROLE = 'AmazonSageMaker-ExecutionRole-20230915T195621'

def init_session():
    sess = sagemaker.Session()

    # sagemaker session bucket -> used for uploading data, models and logs
    # sagemaker will automatically create this bucket if it not exists
    sagemaker_session_bucket=None
    if sagemaker_session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        sagemaker_session_bucket = sess.default_bucket()
    try:
        #role = sagemaker.get_execution_role()
        role = SAGEMAKER_ROLE
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
      version="0.9.3"
    )

    # print ecr image uri
    print(f"llm image uri: {llm_image}")

    return sess, llm_image, role

if __name__ == "__main__":
    init_session()


