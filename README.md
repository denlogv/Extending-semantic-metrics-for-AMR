**TODO:**:
- [ ] AMR-related stuff:
    - [x] Find AMR2TEXT/TEXT2AMR-parser
    - [x] API for computing SMATCH
    - [ ] API for computing S^2MATCH
- [ ] Method:
    - [x] Find constituency/dependency parser
    - [x] Dependency or constituency? --> dependency
    - [ ] Apply Szubert et al. (2018) AMR2DP-mapping
    - [ ] Map text to DP to AMR
- [ ] Datasets:
    - [x] Format STS2016 to a format similar to SICK2014 (s1, s2, theme, score)
    - [ ] MSRP?
    - [ ] PAWS?
- [ ] Analysis:
    - [x] MSE(relatedness, smatch) on SICK2014 


There are currently 2 Jupyter-Notebooks, which showcase the desired functionality of different libraries:

In **semrank.ipynb** you'll find:

    1. AMR-Parser (uses _amrlib_), 
    which parses pairs of sentences in `/datasets/sts/SICK_trial.txt` and for each of 2 columns it produces a .txt-file with AMR graphs in it.

    2.  Function that computes a SMATCH-score for each pair of sentences based on 2 files (`datasets/sts/sents_a_amr.txt`, `datasets/sts/sents_b_amr.txt`) with AMR graphs created in them.
    Then a column with SMATCH-scores is added to SICK, MSE between "relatedness_score" and "f-score" computed, the dataset sorted by MSE, and then everything is saved as `SICK_trial_AMR_SMATCH.tsv`.


In **sentsim_test.ipynb**:

    1. 2 constituency parsers are used (Berkeley Neural Parser – benepar, SuPar) <-- have to decide which one is better

    2. The library `sentence_transformers` is used to compute similarity between NPs


Constituency (dependency) parsers:

1. [SuPar (SOTA)](https://parser.readthedocs.io/en/latest/index.html):

2. [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser)

NB:

BNP uses the old version of _tensorflow_, so it throws an error when using with current versions (>=2.0) of `tensorflow`. The error can be mitigated by changing the folowing code in `ProgramData\Anaconda3\(envs\...)\Lib\site-packages\benepar\base_parser.py`:

`import tensorflow as tf` --> `import tensorflow.compat.v1 as tf`
