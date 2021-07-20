import os
import json
import csv
import numpy as np
import argparse


def get_lookup_source(model, source):



    

    
    

def create_lookup_table(model, source):

    lookupTableData = get_lookup_source(model, source)


if __name__ == '__main__':

    parser = argparse.ArgumentParser

    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--source', type=str, default='gradients', help='gradients/confidence')
    
    args = parser.parse_args()

    create_lookup_table(model, source)
