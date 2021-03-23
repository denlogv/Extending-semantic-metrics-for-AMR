"""
This script converts a .tsv file with following columns:

1) 'sent1'
2) 'sent2'
...
n) ...

to 2 AMR-files, one with all of the first sentences and one with all second sentences.  

Usage example:

python3 tsv2amr.py -i data/SICK2014.tsv -o data/amr/SICK2014_corpus
"""
import amrlib
import argparse
from pathlib import Path


def save_amr(amrs, filepath):
    with open(filepath, 'w') as f:
        for amr in amrs:
            print(amr, file=f, end='\n\n')


def convert_corpus_to_amr(corpus_path, save_path_prefix='processed/corpus'):
    
    with open(corpus_path) as f:
        data = {}
        lines = f.readlines()
        data['a'] = [line.split('\t')[0] for line in lines]
        data['b'] = [line.strip().split('\t')[1] for line in lines]     
    
    stog = amrlib.load_stog_model()
    print('Stog model loaded sucessfully!')
    sents_a, sents_b = data['a'], data['b']
    sents_a_amr = stog.parse_sents(sents_a, add_metadata=True)
    sents_b_amr = stog.parse_sents(sents_b, add_metadata=True) 
    
    Path(save_path_prefix).parent.mkdir(parents=True, exist_ok=True)
                
    save_amr(sents_a_amr, f'{save_path_prefix}_a.amr')
    save_amr(sents_b_amr, f'{save_path_prefix}_b.amr')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='filename to be converted to AMR')
    parser.add_argument('-o', '--output', help='output filename prefix')
    args = parser.parse_args()
    
    convert_corpus_to_amr(args.input, args.output)