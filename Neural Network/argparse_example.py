"""
Example usage:
python -u argparse_example.py --dataset_name mnist --batch_size 120 --M 50 --lr 0.1
"""


import argparse

parser = argparse.ArgumentParser( description='Federated Learning Process' )
parser.add_argument( "--dataset_name", type=str, help="Dataset to use e.x. cifar10, mnist", default='cifar10' )
parser.add_argument( "--batch_size", type=int, help="Bath size", default=200 )
parser.add_argument( "--M", type=int, help="number of units", default=100 )
parser.add_argument( "--client_lr", type=float, help="Learning rate", default=0.1 )

args = parser.parse_args()

dataset_name = args.dataset_name
batch_size = args.batch_size
lr = args.client_lr
M = args.M