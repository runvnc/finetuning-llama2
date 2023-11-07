#!/usr/bin/env python

import argparse
import train

def main():
    parser = argparse.ArgumentParser(
        description='This script loads URLs from a specified file and stores the dataset to a specified S3 bucket path.',
        usage='python train_cli.py <filename> <s3_path>'
    )
    parser.add_argument('filename', type=str, help='The name of the file containing the URLs of the web pages in the dataset. Each URL should be on a new line.')
    parser.add_argument('s3_path', type=str, help='The path to the S3 bucket where the dataset will be stored.')
    args = parser.parse_args()

    # Load URLs from file
    with open(args.filename, 'r') as f:
        urls = [line.strip() for line in f]

    # Call store_url_dataset function
    train.store_url_dataset('meta-llama/Llama-2-7b-hf', args.s3_path, urls)

if __name__ == "__main__":
    main()

