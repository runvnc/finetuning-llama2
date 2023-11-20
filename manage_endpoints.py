#!/usr/bin/env python

import argparse
import boto3
import deploy

def list_endpoints():
    sagemaker_client = boto3.client('sagemaker')
    endpoints = sagemaker_client.list_endpoints(SortBy='CreationTime', SortOrder='Descending')
    for endpoint in endpoints['Endpoints']:
        print(f"{endpoint['EndpointName']} (Status: {endpoint['EndpointStatus']})")

def main():
    parser = argparse.ArgumentParser(description='Manage SageMaker endpoints.')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the 'list' command
    parser_list = subparsers.add_parser('list', help='List all SageMaker endpoints')
    parser_list.set_defaults(func=list_endpoints)

    # Subparser for the 'stop' command
    parser_stop = subparsers.add_parser('stop', help='Stop a SageMaker endpoint')
    parser_stop.add_argument('endpoint_name', type=str, help='The name of the endpoint to stop')
    parser_stop.set_defaults(func=lambda args: deploy.stop_endpoint(args.endpoint_name))

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)

if __name__ == '__main__':
    main()
