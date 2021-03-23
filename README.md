## Introduction:
#### Here is the official repositorium of our team's software project at the **Ruprecht Karl University of Heidelberg**.

Our goal is to extend one of the _AMR_ similarity metrics – **S2Match**, which itself is an extension of **SMatch**.
**S2Match** makes use of _GloVe_-vectors to represent the semantics of concept nodes in _AMR_. This allows for a better triple comparison than it is possible in **SMatch** (because the latter assigns only 1 or 0 for triple similarity.)

Our extension to **S2Match** makes it possible not only to compare one triple with another, but also multiple triples from one AMR with one triple from another (gold), thus allowing for the representation of compositional similarity in _AMR_.

## Methods:
We have implemented 2 ways to achieve this goal.

1. **Merging**  
We found out that most of the compositionality is attributed to cases, in which phrase adjuncts are present. One of most frequent examples is: <br> <br>
$`NP = Det + (Adv) + Adj + N`$ (e.g. _'a very interesting proposition'_) <br> <br>
Such examples are commonly represented using _:mod_-relation in AMR. The idea is to transform an AMR graph in such a way that sentence tokens corresponding to all children nodes of A and A itself ($`= Subtree(A)`$) form a single concept node, which is then to substitute A. <br> $`Subtree(A)`$ is subject to the following conditions:<br>
    - Parent node X has a _:mod_-relation.
    - There are no nodes in $`Subtree(A)`$ (except A) that are used outside of $`Subtree(A) =>`$ has no reentrancies.
    - It corresponds to the complete token span in the sentence.
<br><br>

2. **Remapping** <br>
    One could argue that it may be undesirable to transform graphs in certain cases. To include an option, where this is not necessary we propose a series of steps:
    - Preprocessing graphs to obtain their alignment metadata.
    - Running **S2Match**.
    - Postprocessing **S2Match** in a manner so that a node A in Graph 1 that is currently mapped to _Null_ be remapped according to its alignment metadata. <br> <br>

    These metadata allow us to see whether A has children nodes (similar to Method 1), which together compose a complete token span. We then compute similarity between this token span and a token span that corresponds to a node B in Graph 2.    
    $`sim(A, B)`$, which contributed to the F-Score gets accordingly updated, but only if $`sim_{updated}(A, B) > sim_{original}(A, B)`$. 
    This allows us to have better alignment results, because after the execution of _S2Match_ we apply a postprocessing technique in order to revise the mapping for the unmapped nodes.

## Tools:
Here the structure of our repo is presented. <br>

In the root you can find different scripts which follow the pipeline:

1. Convert a corpus (a _.txt_-file with a SICK dataset or a folder with an STS dataset) to a _.tsv_ (tab-sepated values)-file. **Functionalities:**
    1. sts2tsv.py
    1. sick2tsv
