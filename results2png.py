"""
This script creates a heatmap for smatch resuts. 

Usage examples:

python3 results2png.py --dataset STS --gold data/STS2016_full_fix.tsv \
    --smatch analysis/sts/s2match_glove_results analysis/sts/s2match_sbert_results \
    --output analysis/sts/s2match_modification_results.png
    
python3 results2png.py --dataset SICK --gold analysis/SICK2014_full_scores.tsv \
    --output analysis/sick/s2match_modification_results.png
"""

import sys
import torch
import argparse
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import display
from sentence_transformers import SentenceTransformer, util


def get_paths_names(folders):
    """
    Finds .txt-files with smatch-results in them. 
    Assigns a name to each file according to a number after 'ver' 
    and saves it as a dictionary with pathlib.Paths as keys, 
    and name identifiers to show in a heatmap as values.    
    """
    paths_names = {}
    for folder in folders:
        folder = Path(folder)
        paths  = list(folder.glob('*results*.txt'))
        for path in paths:
            path_str = str(path)
            prefix = 'glove_' if 'glove' in folder.parts[-1] else 'sbert_' if 'sbert' in folder.parts[-1] else 'unknown_'
            if 'orig' in path.parts[-1]:
                name = prefix + 'orig'
                paths_names[path] = name
            else:
                version = path_str[path_str.find('ver') + len('ver')]
                name = prefix + 'v' + version
                paths_names[path] = name
    return paths_names
    

def get_smatch_scores(path, rounded=False):
    """
    Extract scores from lines, which start with 'Smatch score F1' 
    """
    prefix = 'Smatch score F1 '
    scores = []
    with open(path) as f:
        for line in f:
            if line.startswith(prefix):
                if rounded:
                    score = float(round(float(line.split()[-1])))
                else:
                    score = float(line.split()[-1])
                scores.append(score)
    return scores


def update_df(gold, paths_names):
    """
    This functions updates the DataFrame created from the supplied
    .tsv-file that has gold scores in it with s2match analysis results
    and sbert scores from the model 'distilbert-base-nli-stsb-mean-tokens'    
    """

    names   = ['sent1', 'sent2', 'theme', 'sts'] if args.dataset == 'STS' else ['sent1', 'sent2', 'sick']
    usecols = ['sent1', 'sent2', 'sts'] if args.dataset == 'STS' else ['sent1', 'sent2', 'sick']
    
    tsv_df = pd.read_csv(Path(gold), sep='\t', header=0, names=names, usecols=usecols)
    
    for path, name in paths_names.items():
        scores = get_smatch_scores(path)
        if len(scores) > tsv_df.shape[0]:
            print('WARNING, SMATCH RESULTS LENGTH > LENGTH OF THE DATAFRAME. TRUNCATING RESULTS...')
            scores = scores[len(scores)-tsv_df.shape[0]:]
            
        elif len(scores) < tsv_df.shape[0]:
            print('WARNING, SMATCH RESULTS LENGTH < LENGTH OF THE DATAFRAME. TRUNCATING RESULTS...')
            tsv_df = tsv_df[tsv_df.shape[0]-len(scores):, ]
        tsv_df[name] = scores
        
    if args.dataset == 'STS':
        tsv_df['sts'] = tsv_df['sts']/5

    sbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings1 = sbert.encode(tsv_df['sent1'], convert_to_tensor=True)
    embeddings2 = sbert.encode(tsv_df['sent2'], convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    cosine_scores = torch.diagonal(cosine_scores)

    tsv_df['sbert'] = cosine_scores
    return tsv_df

    
def heatm(tsv_df):
    corr_pearson  = tsv_df.corr(method='pearson')
    corr_spearman = tsv_df.corr(method='spearman')
    corrs = [corr_pearson, corr_spearman]

    titles = ['Pearson', 'Spearman']
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    fig.patch.set_facecolor('xkcd:light grey')
    fig.suptitle(f'S2Match Modifications – Results ({args.dataset}):\n', fontsize='xx-large', y=1.05)

    for corr_matrix, title, ax in zip(corrs, titles, axs.flat):
        #plt.figure(figsize=(5,5))
        ax = sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, vmin=0,
                         yticklabels=corr_matrix.columns, annot=True, square=True, 
                         ax=ax)
        ax.invert_yaxis()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title(title)

    fig.savefig(args.output, format='png', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='SICK or STS')
    parser.add_argument('--gold', help='corpus gold scores or a .tsv already filled with all scores')
    parser.add_argument('--smatch', nargs='*', help='folders with s2match results (glove and sbert)')
    parser.add_argument('-o', '--output', help='output .png file with a heatmap of the results')
    args = parser.parse_args()
    
    if args.smatch is not None:
        paths_names = get_paths_names(args.smatch)
        updated_df = update_df(args.gold, paths_names)
        heatm(updated_df)
    else:
        tsv_df = pd.read_csv(Path(args.gold), sep='\t', header=0)
        heatm(tsv_df)
    
    