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
### Root:
In the root directory you can find different scripts which follow the pipeline, which is presented below. For more information on how to combine all the scripts to do the magic, please consult our Jupyter Notebook  [`walkthrough.ipynb`](https://gitlab.com/denlogv/measuring-variation-in-amr/-/blob/master/walkthrough.ipynb). <br>
In order to run this pipeline you'll need to ensure that following criteria are met (it is unfortunate that one has to employ multiple vens to ensure multiple penman versions; alternatively, one could try to install one of the versions directly in the project folder and rename it):

| Script|Prerequisites|
|:----------:|:-------------:|
|`sts2tsv.py` or `sick2tsv.py`|`pandas`|
|`tsv2amr.py`|`amrlib`, `penman>=1.0`| 
|`amr_pipeline.py`|`penman==0.6.2`| <br>
### Pipeline:
1. Convert a corpus (a _.txt_-file with a SICK dataset or a folder with an STS dataset) to a _.tsv_ (tab-sepated values)-file. <br> <br> **Functionalities:** <br> <br>
    - `sts2tsv.py` converts a folder with STS-dataset to a single easily readable _.tsv_-file. <br> <br>
    - `sick2tsv.py` filters a file (.txt file which has a tab-separated-values-layout with 12 columns) with a SICK-dataset to create a .tsv with columns "sent1", "sent2", "sick" (i.e. relatedness-score) <br> <br>
    In our experiments we filtered the dataset to exclude examples, where sentence pairs have entailment label 'CONTRADICTION'
    ```
    Usage examples:

    python3 sick2tsv.py -i datasets/sick/SICK2014_full.txt -o data/SICK2014.tsv --entailment_exclude contradiction
    python3 sts2tsv.py -i datasets/sts/sts2016-english-with-gs-v1.0 -o data/STS2016_full.tsv
    ```
2. Use the created _.tsv_-file to generate 2 _AMR_-files (for each sentence column).  <br> <br>
**Functionalities:** <br> <br>
    - `tsv2amr.py` converts a _.tsv_-file to 2 _AMR_-files
	```
    Usage example:
    
    python3 tsv2amr.py -i data/SICK2014.tsv -o data/amr/SICK2014_corpus
    ```
3. Create alignment files for each _AMR_-file using the _AMR2Text_-alignment tool (TAMR/JAMR) presented in [**HIT-SCIR CoNLL2019 Unified Transition-Parser**](https://github.com/DreamerDeo/HIT-SCIR-CoNLL2019).<br> <br>
**Functionalities:** <br> <br>
    - `amr_pipeline.py` converts a _.tsv_-file to 2 _AMR_-files
	```
    Usage example:
    
    python3 AMR2text/amr_pipeline.py -o data/amr/STS2016_corpus
    ```
4. Analyse the alignment files and either transform the _AMR_-graphs according to **Method 1** or add metadata to them according to **Method 2 **. <br> <br>
**Functionalities:** <br> <br>
    - `AMRAnalysis.py` takes 1 or 2 _AMR2Text_-alignment-files and either transforms the graphs, or adds metadata to these file. Outputs 1 or 2 _AMR_-files.
	```
    Usage example:
    
    python3 AMRAnalysis.py -i data/amr/SICK2014_corpus_a_aligned.mrp data/amr/SICK2014_corpus_b_aligned.mrp --output_prefix analysis/sick/SICK2014 --extended_meta
    ```
5. Run **S2Match** on  the resulting _AMR_-files.
6. Evaluate by computing _Spearman rank_ and _Pearson correlation coefficients_ + Visualise the results. <br> <br>
**Functionalities:** <br> <br>
    - for steps 5 and 6 please consult our Jupyter Notebook  [`walkthrough.ipynb`](https://gitlab.com/denlogv/measuring-variation-in-amr/-/blob/master/walkthrough.ipynb). Standalone scripts will be added soon. 