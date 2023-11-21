import sagemaker
import boto3
import time
import json

from sagemaker.huggingface import HuggingFaceModel

import init_sagemaker

def stop_endpoint(endpoint_name):
    """
    Stops a SageMaker endpoint and deletes the associated resources.

    :param endpoint_name: The name of the endpoint to stop.
    """
    sagemaker_client = boto3.client('sagemaker')
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} has been stopped and deleted.")
 
def deploy_tgi_model_from_url(model_data, endpoint_name, instance_type = "ml.g5.12xlarge",
          model_id="/opt/ml/model", number_of_gpu = 4, health_check_timeout = 300):

    (sess, llm_image, role) = init_sagemaker.init_session()
    
    TGI_config = {
        'HF_MODEL_ID': model_id, 
        'SM_NUM_GPUS': json.dumps(number_of_gpu),
        'MAX_INPUT_LENGTH': json.dumps(1024), 
        'MAX_TOTAL_TOKENS': json.dumps(2048),
        'MAX_BATCH_TOTAL_TOKENS': json.dumps(4096),
        'QUANTIZE': json.dumps("bitsandbytes"),
        'PREDIBASE_MODEL_BUCKET': json.dumps(sess.default_bucket())
        }
    # 'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
	
    llm_model = HuggingFaceModel(
	  role=role,
	  image_uri=llm_image,
	  model_data=model_data,
	  env=TGI_config
	)
    print( { 'role': role, 'image_uri': llm_image, 'model_data': model_data, 'env': TGI_config } )    

    print( { 'endpoint_name': endpoint_name, 'instance_type': instance_type })
    llm = llm_model.deploy(
	  endpoint_name=endpoint_name,
	  initial_instance_count=1,
	  instance_type=instance_type,
	  #volume_size=100, # If using an instance with local SSD storage, volume_size must be None, e.g. p4 but not p3
	  container_startup_health_check_timeout=health_check_timeout, # 10 minutes to be able to load the model
	)

