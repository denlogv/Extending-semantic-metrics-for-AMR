# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:29:46 2021

@author: Denis Logvinenko
"""
with open('corpus_a.txt') as inp1, \
     open('corpus_b.txt') as inp2, \
     open('corpus.tsv', 'w') as out:
         
    lines1 = [line.strip() for line in inp1.readlines()]
    lines2 = inp2.readlines()
    zipped = zip(lines1, lines2)
    
    for l1, l2 in zipped:
        out.write(f'{l1}\t{l2}')