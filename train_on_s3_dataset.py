#!/usr/bin/env python

import argparse
import train

def main():
    parser = argparse.ArgumentParser(
        description='This script fine-tunes an LLM on the specified dataset in s3.',
        usage='./train_on_s3_dataset.py <s3_path>'
    )
    parser.add_argument('s3_partial_path', type=str, 
      help='The partial path name to the S3 bucket where the dataset will be stored (off of the default bucket).')
    args = parser.parse_args()

    train.fine_tune(args.s3_partial_path, 'meta-llama/Llama-2-7b-hf')

if __name__ == "__main__":
    main()

