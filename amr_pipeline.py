"""
This script does the following:
    
    1) It converts .amr-files to MRP-format
    2) It runs the AMR2Text-alignment tool on the MRP-corpora.

Usage example:

amr_pipeline.py -t AMR2Text -o data/amr/STS2016_corpus
"""

"""
Shell commands from different tools, on which this script relies:
mtool:
    AMR to MRP: python3 main.py --read amr --write mrp <read_filename>.amr <out_filename>.mrp
    MRP to dot: python3 main.py --read mrp --write dot <read_filename>.mrp <out_filename>.dot
    dot to pdf: dot -Tpdf amr_test.dot > amr_test.pdf

bash/amr_preprocess.sh:
    bash amr_preprocess.sh <filename>.mrp glove.840B.300d.w2v.txt

python3 toolkit/mtool20/main.py --read mrp --write dot processed/corpus_a.mrp processed/corpus_a.dot
dot -Tpdf processed/corpus_a.dot > processed/corpus_a.pdf
"""
import os
import sys
import argparse
import subprocess, shlex
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', help='output filename prefix')
parser.add_argument('-t', '--tooldirectory', default='AMR2Text', help='path to directory of HIT-SCIR-CoNLL2019 AMR2Text-tool')
args = parser.parse_args()


if __name__ == '__main__':
    
    os.chdir(args.tooldirectory)
    
    commands = [
        f"python3 toolkit/mtool20/main.py --read amr --write mrp {'..//'+args.output}_a.amr {'..//'+args.output}_a.mrp",
        f"python3 toolkit/mtool20/main.py --read amr --write mrp {'..//'+args.output}_b.amr {'..//'+args.output}_b.mrp",
        f"bash bash/amr_preprocess.sh {'..//'+args.output}_a.mrp bash/glove.840B.300d.w2v.txt",
        f"bash bash/amr_preprocess.sh {'..//'+args.output}_b.mrp bash/glove.840B.300d.w2v.txt"
    ]
    
    # FOR DEBUGGING ONLY. COMMENT IT OUT OR DELETE IT FOR NORMAL USE!!! 
    #commands = [f"bash bash/amr_preprocess.sh {'..//'+args.output}_b.mrp bash/glove.840B.300d.w2v.txt"]
    
    for command in commands:
        print(command, '\nis being executed...','\n')
        res = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT, text=True)
        for line in res.stdout.split('\n'):            
            if line:
                print(line)
    
    os.chdir('..')
    