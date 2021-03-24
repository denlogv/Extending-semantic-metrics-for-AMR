"""
This script takes one or two AMR2Text-alignment-files and either makes transformations on the graphs 
(merges subtrees with certain relations (e.g ':mod') according to some rules), or adds
alingment metadata to these files.

Usage example:

python3 AMRAnalysis.py -i data/amr/SICK2014_corpus_a_aligned.mrp data/amr/SICK2014_corpus_b_aligned.mrp --output_prefix analysis/sick/SICK2014 --extended_meta
"""

import penman
from penman import layout
from penman.graph import Graph
from penman.transform import reify_attributes

import re
import json
import argparse
from pathlib import Path
from collections import defaultdict


def save_corpus(path, amr_analysis, concatenation=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        if concatenation:
            if not amr_analysis.graphs_concat_rel:
                amr_analysis.concat_rel()
            for amr_id, (g, g_concat) in amr_analysis.graphs_concat_rel.items():
                meta_block = amr_analysis.info_dict[amr_id]['meta']
                print(meta_block, file=f)
                pprint(g_concat, file=f)

        else:
            for amr_id in amr_analysis.info_dict:
                meta_block = amr_analysis.info_dict[amr_id]['meta']
                print(meta_block, file=f)
                pprint(amr_analysis.info_dict[amr_id]['amr_string'], file=f)


def pprint(l, reified=False, **args):
    if isinstance(l, dict):
        print('Key\tValue')
        for k, v in l.items():            
            print(f'{k}\t{v}', **args)
            
    elif isinstance(l, list) or isinstance(l, tuple) or isinstance(l, set):
        for el in l:
            print(el, **args)
            
    elif isinstance(l, penman.Graph):
        if reified:
            l = penman.encode(l)
            l = reify_rename_graph_from_string(l)
        print(penman.encode(l), **args)
        
    elif isinstance(l, penman.Tree):
        if reified:
            l = penman.format(l)
            l = reify_rename_graph_from_string(l)
            print(penman.encode(l), **args)
        else:
            print(penman.format(l), **args)
        
    elif isinstance(l, str):
        if reified:
            l = reify_rename_graph_from_string(l)
            print(penman.encode(l), **args)
        else:
            print(penman.format(penman.parse(l)), **args)
            
    else:
        raise ValueError('Unknown type')
    print(**args)
    
class AMRAnalysis:
    def __init__(self, amr2text_alingnment_path, keep_meta=True, 
                 extended_meta=False, concat_rel=False):
        self.amr2text_alingnment_path = amr2text_alingnment_path        
        self.keep_meta = keep_meta
        if extended_meta:
            self.keep_meta = self.extended_meta = True
        else:
            self.extended_meta = False
        self.info_dict = {}
        self.graphs_concat_rel = {}
        if concat_rel:
            self.concat_rel()
        else:
            self.extract_info()
    
    @staticmethod
    def reify_rename_graph_from_string(amr_string):
    
        g1 = reify_attributes(penman.decode(amr_string))
        t1 = layout.configure(g1)
        t1.reset_variables(fmt='MRPNode-{i}')
        g1 = layout.interpret(t1)

        return g1
    
    @staticmethod
    def alignment_labels2mrp_labels(amr_string):
        """Currently works only on reified graphs"""

        amr_graph = AMRAnalysis.reify_rename_graph_from_string(amr_string)
        epidata, triples = amr_graph.epidata, amr_graph.triples
        cur_label, popped = '0', False
        labels_dict = {cur_label:amr_graph.top}
        for triple in triples:        
            cur_node = triple[0]        
            epi = epidata[triple]
            if epi and isinstance(epi[0], penman.layout.Push):
                cur_node = epi[0].variable
                if not popped:
                    cur_label += '.0'
                labels_dict[cur_label] = cur_node
                popped = False            
            elif epi and isinstance(epi[0], penman.layout.Pop):
                pops_count = epi.count(epi[0])
                split = cur_label.split('.')
                if popped: 
                    split = split[:len(split)-pops_count] 
                else:
                    split = split[:len(split)-pops_count+1]
                split[-1] = str(int(split[-1])+1)
                cur_label = '.'.join(split)
                popped = True

        return labels_dict, amr_graph
    
    @staticmethod
    def get_alignments_dict_from_string(alignments_string, alignment_pattern, toks, labels_dict):
        """
        Somehow the alingnments string in 'new_alinged' does not contain
        all aligned nodes that are specified below ¯\_(ツ)_/¯ 
        """
        matches = re.match(alignment_pattern, alignments_string)
        if not matches:
            raise ValueError(f'Alignments string "{alignments_string}" has wrong format!\nCould not find alignments.')
        alignments = matches.group(1).split()
        alignments_dict = {}

        for alignment in alignments:
            parts = alignment.split('|')
            token_span = parts[0]
            #indices = span.split('-')
            #token_span = ' '.join(toks[int(indices[0]):int(indices[1])])
            nodes = parts[1].split('+')
            nodes = [labels_dict[node] for node in nodes]
            for node in nodes:
                alignments_dict[node] = token_span
        return alignments_dict
    
    @staticmethod
    def get_alignments_dict(nodes_block, labels_dict, alignments_with_toks=False, toks=None):
        """
        This function deals with the problem that was found while using the 
        function above
        """
        nodes_block = [spl_line for spl_line in nodes_block if len(spl_line) == 3]
        alignments_dict = {}
        for spl_line in nodes_block:
            node = spl_line[0]
            node = labels_dict[node] # '0.0.0' --> 'MRPNode2'
            token_span = spl_line[2]
            if alignments_with_toks:
                start_idx, end_idx = token_span.split('-')
                token_span = ' '.join(toks[int(start_idx):int(end_idx)])
            alignments_dict[node] = token_span
            
        return alignments_dict

    def extract_info(self, alignments_with_toks=False):    
        with open(self.amr2text_alingnment_path) as f:
            amrs = f.read().strip().split('\n\n')
            amrs = [amr.split('\n') for amr in amrs]

        alignment_pattern = re.compile(r'# ::alignments\s(.+?)\s::')

        for amr_analysis in amrs:
            amr_id = amr_analysis[0].split()[-1]

            toks = amr_analysis[2].split()[2:] # first 2 tokens are: '# ::tok'
            toks = [tok.lower() for tok in toks]

            amr_string = amr_analysis[-1]
            labels_dict, amr_graph = AMRAnalysis.alignment_labels2mrp_labels(amr_string)

            alignments_string = amr_analysis[3]
            nodes_block = [line.split()[2:] for line in amr_analysis if line.startswith('# ::node')] # first 2 tokens are: '# ::node'
            try:
                # function below works well, but the alignments string doesn't contain all alignments, 
                # so a new function has to be defined
                #alignments_dict = AMRAnalysis.get_alignments_dict_from_string(alignments_string, alignment_pattern, toks, labels_dict)
                alignments_dict = AMRAnalysis.get_alignments_dict(nodes_block, labels_dict, alignments_with_toks, toks)
                alignments_dict = defaultdict(lambda: None, alignments_dict)
            except KeyError as e:
                print(amr_id)
                pprint(amr_string, reified=True)
                pprint(labels_dict)
                raise e

            self.info_dict[amr_id] = {'amr_string':penman.encode(amr_graph), \
                                      'toks':toks, \
                                      'alignments_dict':alignments_dict, \
                                      'labels_dict':labels_dict, \
                                      'amr_graph':amr_graph}
            if self.keep_meta:
                meta = amr_analysis[:3] # save '# ::id', '# ::snt' fields
                meta = '\n'.join(meta)
                self.info_dict[amr_id]['meta'] =  meta
            if self.extended_meta:
                labels_dict_string = json.dumps(labels_dict)
                alignments_dict = json.dumps(alignments_dict)
                self.info_dict[amr_id]['meta'] +=  f'\n# ::labels_dict {labels_dict_string}\n# ::alignments_dict {alignments_dict}'
                
        return self
    
    @staticmethod
    def find_below(labels_dict):
        """
        Finds nodes below a certain node using a dictionary of the following form
        (located in 'info_dict[amr_id]['labels_dict']'):
        
        Key Value
        0 MRPNode-0
        0.0 MRPNode-1
        0.0.0 MRPNode-2
        0.0.0.0	MRPNode-3
        0.0.0.0.0 MRPNode-4
        0.0.0.0.1 MRPNode-5
        0.0.1 MRPNode-6
        0.0.1.0 MRPNode-7
        
        Returns a dict where the key is the node label (e.g 'MRPNode-2') and
        the value is a list with all nodes represented as strings below it.
        """
        nodes_below_dict = defaultdict(list)
        for key, value in labels_dict.items():
            for k, v in labels_dict.items():
                if k.startswith(key) and len(k) > len(key):
                    nodes_below_dict[value].append(v)
        return nodes_below_dict
    
    @staticmethod
    def full_span(subtree_token_spans):
        """
        Takes a list of token spans of a whole subtree
        and checks, if there are gaps. 
        
        Returns a list of indices if a token span is full, else False.
        """
        toks_indices = set()
        for token_span in subtree_token_spans:
            spl = token_span.split('-')
            i1, i2 = int(spl[0]), int(spl[1])
            indices = set(range(i1, i2))            
            toks_indices.update(indices)
        if not toks_indices:
            return None
        minimum, maximum = min(toks_indices), max(toks_indices)
        toks_indices = sorted(list(toks_indices))
        if toks_indices == list(range(minimum, maximum+1)):
            return toks_indices
        return None
    
    def concat_rel(self, rel=':mod'): 
        if not self.info_dict:
            self.extract_info()
        self.graphs_concat_rel = {}
        
        # ONLY FOR DEBUGGING CERTAIN IDS!!!
        # DELETE FOR NORMAL USE!!!
        #self.info_dict = {k:v for k, v in self.info_dict.items() if k == '3'}
        
        for amr_id in self.info_dict:
            triples_filtered = []
            g = self.info_dict[amr_id]['amr_graph']
            toks = self.info_dict[amr_id]['toks']
            alignments_dict = self.info_dict[amr_id]['alignments_dict']
            nodes_below_dict = AMRAnalysis.find_below(self.info_dict[amr_id]['labels_dict'])
            instances_dict = defaultdict(lambda: None, {node:concept for node, _, concept in g.instances()})
            reentrancies = defaultdict(lambda: None, g.reentrancies())
            
            changed_instances = {}
            nodes_to_delete = []
            epidata = {}
            
            for triple in g.triples:
                if triple[0] not in nodes_to_delete and triple[2] not in nodes_to_delete:
                    if triple[1] == rel:
                        invoked = triple[0]
                        nodes_below_invoked = nodes_below_dict[invoked]
                        nodes_below_invoked_with_invoked = nodes_below_invoked + [invoked]
                        instances_below_invoked = [instances_dict[node] for node in nodes_below_invoked]
                        
                        span = [alignments_dict[node] for node in nodes_below_invoked_with_invoked if alignments_dict[node]]
                        subtree_token_span = AMRAnalysis.full_span(span)
                        reentrancies_below_invoked = any([reentrancies[node] for node in nodes_below_invoked])
                        
                        if subtree_token_span and not reentrancies_below_invoked:
                            merged = [toks[i] for i in subtree_token_span]
                            num_nodes_in_subtree = len(nodes_below_invoked_with_invoked)
                            changed_instances[invoked] =  '_'.join(merged) + '_' + str(num_nodes_in_subtree)                           
                            nodes_to_delete += nodes_below_invoked
                            continue
                            
                    epidata[triple] = g.epidata[triple]
                    triples_filtered.append(triple)
            
            for i in range(len(triples_filtered)):
                n, r, c = triples_filtered[i]
                old_tuple = (n, r, c)
                if n in changed_instances and r == ':instance':
                    new_tuple = (n, r, changed_instances[n])
                    triples_filtered[i] = new_tuple
                    epidata = {(k if k != old_tuple else new_tuple):(v if k != old_tuple else v+[penman.layout.Pop()]) 
                               for k, v in epidata.items()}
            
            new_g = Graph(triples=triples_filtered, epidata=epidata)
            new_t = layout.configure(new_g)
            new_t.reset_variables(fmt='MRPNode-{i}')
            new_g = layout.interpret(new_t)
            self.graphs_concat_rel[amr_id] = (g, new_g)
            
            collapsed_instance_nodes = len(nodes_to_delete)
            if self.keep_meta:
                self.info_dict[amr_id]['meta'] += f'\n# ::collapsed instance nodes {collapsed_instance_nodes}'
            #else:
            #    self.info_dict[amr_id]['meta'] = f'# ::collapsed instance nodes {collapsed_instance_nodes}'
            
        return self

    
def do_all_stuff(args):
        
    if (not args.concat_rel) and (not args.extended_meta):
        output_suffix = 'reif'
      
    elif args.concat_rel and args.extended_meta:
        output_suffix = 'concat_ext'
        
    elif args.concat_rel:
        output_suffix = 'concat'
    
    else:
        output_suffix = 'reif_ext'        
    
    print(f'Input parameters: concat_rel={args.concat_rel}, extended_meta={args.extended_meta}')
    
    for i, f in enumerate(args.input[:2]):
        amr_analysis = AMRAnalysis(f, concat_rel=args.concat_rel,
                                   extended_meta=args.extended_meta)
        
        save = Path(f'{args.output_prefix}_corpus_{chr(97+i)}_{output_suffix}.amr')
        
        print(f'File: "{str(save)}" was sucessfully saved!')
        save_corpus(save, amr_analysis, concatenation=args.concat_rel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', 
                        help='path(s) of the amr2text alignment file')
    parser.add_argument('--extended_meta', action='store_true', default=False, 
                        help='defines whether alignment meta has to be added to AMRs')
    parser.add_argument('--concat_rel', action='store_true', default=False, 
                        help='defines whether AMR graphs have to be transformed according to their token alignments')
    parser.add_argument('--output_prefix', default='analysis/sts/STS2016',
                        help='defines the prefix of the output file(s), e.g. if == STS2016 -> STS2016_corpus_(a|b)_(reif|ext|concat).amr')
    args = parser.parse_args()
    
    do_all_stuff(args)

