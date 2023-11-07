import argparse
import train

def main():
    parser = argparse.ArgumentParser(description='Load URLs and store dataset.')
    parser.add_argument('filename', type=str, help='File containing URLs')
    parser.add_argument('s3_path', type=str, help='S3 bucket path')
    args = parser.parse_args()

    # Load URLs from file
    with open(args.filename, 'r') as f:
        urls = [line.strip() for line in f]

    # Call store_url_dataset function
    train.store_url_dataset('meta-llama/Llama-2-7b-hf', args.s3_path, urls)

if __name__ == "__main__":
    main()

