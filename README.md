# Introduction

This repository contains the scripts and released lexicons described in the paper [A Simple Method for Unsupervised Bilingual Lexicon Induction for Data-Imbalanced, Closely Related Language Pairs](https://arxiv.org/abs/2305.14012). We present a fast and simple method for Bilingual Lexicon Induction for closely related languages, relying on good quality contextual embeddings in only one of the languages. This scenario is especially applicable to the so-called ''Hindi Belt'', containing 40+ languages and dialects, spoken by 100M+ people in North India. 

In this work, we use Hindi as our high-resource language, and automatically create lexicons for Bhojpuri, Magahi, Awadhi, Braj, and Maithili. 

# Data

## Released lexicons
Find our (automatically created) lexicons in ``bli/lexicons/<lang>/``, for ``lang``: ``awa``, ``bho``, ``bra``, ``mai``, ``mag``.

Their sizes are as follows:

| Language      | Lexicon   | 
|---------------|-----------|
| Awadhi        | 10462     |
| Bhojpuri      | 21983     |
| Braj          | 10760     | 
| Magahi        | 30784     | 
| Maithili      | 12069     | 
     
## Evaluation lexicons

Our created/compiled evaluation lexicons can be found in ``get_eval_lexicon/lexicons/target2hin/``

### Bhojpuri and Magahi
We create silver lexicons for evaluation using word alignments from parallel data as described in our paper. We collect 2469 and 3359 entries for Bhojpuri and Magahi respectively. Our code for the same can be found in ``get_eval_lexicon/``.

### Marathi and Nepali
These lexicons were directly extracted from [IndoWordNet](https://www.cfilt.iitb.ac.in/indowordnet/) using their [Python API](https://www.cse.iitb.ac.in/~pb/papers/gwc18-pyiwn.pdf). In particular, we equated entries in parallel synsets. These lexicons contain 35000 and 22000 entries for Marathi and Nepali respectively.

# Running our method

## Basic
Here is a description of parameters to be set.

* ``TARGET_FILE_PATH``: Path to target language monolingual data, one sentence per line
* ``hf_model_name``: Can be a pretrained HuggingFace model identifier or a path to a locally trained mask filling model.
* ``OUTDIR``: Directory where lexicons will be saved
* ``threshold (default:0.5)``: minimum normalized edit distance (NED) between a word and translation candidate.
* ``iterations (default:3)``: maximum number of times a single (sentence, word) instance will be processed
* ``batch_size``: number of sentences processed at once by the mask filling pipeline.
* ``lang``: Target language code, only required for naming purposes

Run our script as follows:

```
python bli/scripts/basic.py --TARGET_FILE_PATH $TARGET_FILE_PATH --hf_model_name $HF_MODEL_NAME --OUTDIR $OUTDIR 
--threshold $threshold --iterations $iterations --batch_size $batch_size --lang $lang 
```

## Rulebook
For the ``Rulebook`` method, provide an extra path 
* ``PARAMS_OUTDIR``: Directory where learnt transition probabilities (LRL --> HRL) are saved.


Run our script as follows:

```
python bli/scripts/rulebook.py --TARGET_FILE_PATH $TARGET_FILE_PATH --hf_model_name $HF_MODEL_NAME --OUTDIR $OUTDIR 
--threshold $threshold --iterations $iterations --batch_size $batch_size --lang $lang --PARAMS_DIR $PARAMS_DIR
```

## Evaluation

Set the following parameters:
* ``GOLD_LEXICON_PATH``: Path to evaluation lexicon, formatted as JSON, with associated scores.
* ``PRED_LEXICON_PATH``: Path to predicted lexicon, formatted as JSON, with associated scores.
* ``top_k_pred (default:2)``: Number of top predictions to consider per word (higher score is assumed to be better).
* ``top_k_gold (default:None)``: If set, number of top equivalents to consider as correct from the gold lexicon. If not set, all given equivalents are considered correct.

Run our script as follows:
```
python bli/evaluate_lexicons.py --GOLD_LEXICON_PATH $GOLD_LEXICON_PATH --PRED_LEXICON_PATH $PRED_LEXICON_PATH
```

