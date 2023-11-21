#!/usr/bin/env python

import argparse
import dataset

def main():
    parser = argparse.ArgumentParser(
        description='This script loads documents from a specified dir, extracts the text from those documents, and stores the dataset to a specified S3 bucket path.',
        usage='./upload_docs_dataset_to_s3.py <url_list_filename> <s3_path_name>'
    )
    parser.add_argument('docs_path', type=str, help='The path to the documents.')
    parser.add_argument('s3_path_name', type=str, help='The partial name in S3 where the dataset will be stored (in the default bucket).')
    args = parser.parse_args()

    dataset.store_url_dataset('meta-llama/Llama-2-13b-chat-hf', args.docs_path, args.docs_path)

if __name__ == "__main__":
    main()

