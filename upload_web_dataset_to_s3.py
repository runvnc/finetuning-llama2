#!/usr/bin/env python

import argparse
import dataset

def main():
    parser = argparse.ArgumentParser(
        description='This script loads URLs from a specified file, extracts the text from those web pages, and stores the dataset to a specified S3 bucket path.',
        usage='./upload_web_dataset_to_s3.py <url_list_filename> <s3_path_name>'
    )
    parser.add_argument('url_list_filename', type=str, help='The name of the file containing the URLs of the web pages in the dataset. Each URL should be on a new line.')
    parser.add_argument('s3_path_name', type=str, help='The partial name in S3 where the dataset will be stored (in the default bucket).')
    args = parser.parse_args()

    with open(args.url_list_filename, 'r') as f:
        urls = [line.strip() for line in f]

    dataset.store_url_dataset('meta-llama/Llama-2-13b-chat-hf', args.s3_path_name, urls)

if __name__ == "__main__":
    main()

