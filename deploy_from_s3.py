#!/usr/bin/env python

import argparse
import deploy

def main():
    parser = argparse.ArgumentParser(
        description='This script deploys a TGI LLM model from a full S3 URL to create a Sagemaker inference endpoint.',
        usage='./deploy_from_s3.py <s3_full_url> <endpoint_name>'
    )
    parser.add_argument('s3_full_url', type=str, 
      help='The full URL to the model in S3.')
    parser.add_argument('endpoint_name', type=str,
      help='The inference endpoint name.')

    args = parser.parse_args()

    deploy.deploy_tgi_model_from_url(args.s3_full_url, args.endpoint_name)

if __name__ == "__main__":
    main()

