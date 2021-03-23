"""
This script converts the whole folder with a STS dataset to a single
easily readable .tsv file with following columns:

1) 'sent1'
2) 'sent2'
3) 'theme'
4) 'score'

Usage example:

python3 sts2tsv.py -i datasets/sts/sts2016-english-with-gs-v1.0 -o data/STS2016_full.tsv
"""


import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input STS folder')
parser.add_argument('-o', '--output', help='output .tsv corpus file')
args = parser.parse_args()


if __name__ == '__main__':
    #sts = Path.cwd()/'datasets'/'sts'/'sts2016-english-with-gs-v1.0'
    
    sts = Path(args.input)
    
    sts_ds_all   = list(sts.glob('STS*.*.txt'))
    sts_ds_input = sorted([p for p in sts_ds_all if p.suffixes[0] == '.input'])
    sts_ds_gold  = sorted([p for p in sts_ds_all if p.suffixes[0] == '.gs'])
    sts_ds_input_gold = list(zip(sts_ds_input, sts_ds_gold))

    sts_ds = defaultdict(list)
    get_theme = lambda p: p.suffixes[1][1:]

    for file_input, file_gold in sts_ds_input_gold:
        theme = get_theme(file_input)
        with open(file_gold) as gold: 
            scores = [np.nan if not line.strip() else int(line.strip()) for line in gold.readlines()]
            sts_ds['score'] += scores
            sts_ds['theme'] += [theme]*len(scores)

        with open(file_input) as inp:
            for line in inp:
                if line:
                    sent1, sent2 = line.split('\t')[:2]
                    sts_ds['sent1'].append(sent1); sts_ds['sent2'].append(sent2)
    
    sts_df = pd.DataFrame(sts_ds)[['sent1', 'sent2', 'theme', 'score']]
    output_path = Path(args.output)
    Path(output_path.parent).mkdir(parents=True, exist_ok=True)
    sts_df = sts_df.loc[(sts_df['score'].notnull()) & (sts_df['theme'] != 'postediting')]
    sts_df.sort_values(by=['theme', 'score'], ascending=False).to_csv(output_path, sep='\t', index=False)
    print(f'File "{args.input}" has been successfully converted to "{args.output}"!')