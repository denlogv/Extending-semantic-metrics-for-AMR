# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:18:39 2021

@author: Denis Logvinenko
"""
import amrlib
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='filename to be converted to AMR')
parser.add_argument('-o', '--output', help='output filename prefix')
args = parser.parse_args()


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
    convert_corpus_to_amr(args.input, args.output)