"""
This script filters a file (.txt file which has a tab-separated-values-layout with 12 columns) with a SICK-dataset to create a .tsv with following columns:

1) 'sent1'
2) 'sent2'
3) 'sick'

In our experiments we filtered the dataset to exclude examples, where sentence pairs have
entailment label 'CONTRADICTION'

Usage example:

python3 sick2tsv.py -i datasets/sick/SICK2014_full.txt -o data/SICK2014.tsv --entailment_exclude contradiction
"""


import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input SICK file')
parser.add_argument('-o', '--output', help='output .tsv corpus file')
parser.add_argument('--entailment_exclude', default=None)
args = parser.parse_args()


if __name__ == '__main__':
    colnames = ['c1', 'sent1', 'sent2', 'entailment', 'sick', \
                'c6', 'c7','c8', 'c9', 'c10', 'c11', 'c12']

    df = pd.read_csv(Path(args.input), sep='\t', header=0, 
                     usecols=['sent1', 'sent2', 'sick', 'entailment'], 
                     names=colnames)

    df['sick'] = (df['sick'] - 1)/4
    if args.entailment_exclude is not None:
        df = df[df['entailment'] != (args.entailment_exclude).upper()]
    df.drop(columns=['entailment'], inplace=True)
    df.sort_values(by=['sick'], ascending=False).to_csv(Path(args.output), sep='\t', index=False)
    print(f'File "{args.input}" has been successfully converted to "{args.output}"!')