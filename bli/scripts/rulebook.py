# %%

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, BertForMaskedLM
from transformers import pipeline
from collections import defaultdict, Counter
import editdistance
import math, random
import json
import os, sys
import atexit
sys.path.append("../../")

from utils import get_tokenizers
from basic import BLIBasic, PipeInput, get_args
from tqdm import tqdm
import torch

import os
import argparse
import math
import json
from collections import defaultdict, Counter
import pandas as pd
import editdistance
import Levenshtein as lv
import copy
import random
import string


class BLIRulebook(BLIBasic):

    def __init__(self, TOKENIZER_INPATH=None, MODEL_INPATH=None, HF_MODEL_NAME=None, \
        TARGET_FILE_PATH=None, OUTPATH="lexicon.json", batch_size=500, iterations=3, threshold=0.5, PARAMS_DIR = None) -> None:
        super().__init__(TOKENIZER_INPATH, MODEL_INPATH, HF_MODEL_NAME, TARGET_FILE_PATH, OUTPATH, batch_size, iterations, threshold)            
        self.prev_trans_matrix = defaultdict(lambda:defaultdict(lambda:0))
        self.trans_matrix = defaultdict(lambda:defaultdict(lambda:0))
        self.totals = defaultdict(lambda:1)
        
        self.seen = defaultdict(lambda:set())
        self.null_char = "NULL"
        
        self.init_same_char_counts = 1
        self.init_total = 2
        
        self.training = False
        
        self.sem_nns = dict()
        self.PARAMS_DIR = PARAMS_DIR

    def initialize_trans_matrix(self, load_pretrained = False, type = "uniform"):
        '''Initialize transition matrix and totals matrix, including insertions and deletions'''

        if load_pretrained:
            print("Loading pretrained...")
            self.load_model_params()
            return

        print("Intializing TRANSITION MATRIX...")
        #Get source and target chars 

        def get_source_target_chars():

            # all_source = Counter("".join(sent["text"] for sent in self.rem_sentences))
            # good_chars = {ch for ch in all_source if ch not in string.punctuation and \
            #     not ch.isspace() and ch!="।"}
            source_chars = set("".join(sent["text"] for sent in self.rem_sentences)) 
            source_chars = {ch for ch in source_chars if self.is_dev(ch)}
            source_chars.add(self.null_char)

            tokenizer_vocab = set(self.tokenizer.vocab.keys())
            target_chars = set("".join(word for word in tokenizer_vocab))
            target_chars = {ch for ch in target_chars if self.is_dev(ch)}
            target_chars.add(self.null_char)

            return source_chars, target_chars

        source_chars, target_chars = get_source_target_chars()
        total_targets = len(target_chars)

        for source in source_chars:
            for target in target_chars:
                if source == target:
                    self.trans_matrix[source][target] = self.init_same_char_counts
                    continue

                if source in target_chars:
                    self.trans_matrix[source][target] = \
                    (self.init_total-self.init_same_char_counts)/(total_targets-1)

                else:
                    self.trans_matrix[source][target] = \
                    self.init_total/total_targets

        print("Intialized TRANSITION MATRIX of dimensions {}x{}".format(len(source_chars)+1, len(target_chars)+1))


    # def initialize_model_params(self, iterations, batch_size, updates, \
    #                             cand_source_words, cand_target_words, \
    #                             load_pretrained_model = False, PARAMS_DIR = None):
    #     '''Initializes training params'''


    #     self.batch_size = batch_size
    #     self.iterations = iterations
    #     self.updates = updates

    #     if load_pretrained_model:
    #         print("Loading pretrained...")
    #         self.load_model_params(PARAMS_DIR)
    #     else:
    #         self.initialize_trans_matrix(cand_source_words, cand_target_words)

    #     #self.totals and self.seen are already correctly initialized



    def op2chars(self, op, source, target):
        '''Returns chars based on levenshtein op'''

        if op[0] == "replace":
            char1 = source[op[1]]
            char2 = target[op[2]]
        if op[0] == "insert":
            char1 = self.null_char
            char2 = target[op[2]]
        if op[0] == "delete":
            char1 = source[op[1]]
            char2 = self.null_char
        if op[0] == "retain":
            char1 = source[op[1]]
            char2 = source[op[1]]

        return char1, char2


    def augmented_ops(self, source, target):
        '''Returns minimal ed ops but also char retentions'''
        # lv.editops returns (op, sidx, tidx) such that op is performed on target[tidx] 
        # at source[sidx] for replace and insert and on source[sidx] for delete
        # Therefore all chars at sidx that were "replaced at" are not retained.
        # Since insertions are "hypothetical" i.e. target[tidx] is not really at sidx,
        # source[idx] should not be eliminated for retention
        # For deletions, it's the other way around, i.e. source[sidx] is deleted
        # And so we eliminate source[sidx] for retention as well.

        ops = lv.editops(source, target)
        bad_sidxs = {sidx for (op, sidx, _) in ops if op != "insert"}
        ret_idxs = [("retain", sidx) for sidx in range(len(source)) if sidx not in bad_sidxs]
        return ops + ret_idxs
        ### If we DO NOT WANT TO ADD RETENTIONS, uncomment the following
        # return ops 



    def update_params(self, pair):
        '''Update matrix counts given a new observation'''
        (source, target) = pair
        ops = self.augmented_ops(source, target)

        for op in ops:
            char1, char2 = self.op2chars(op, source, target)
            ### NOT UPDATING WEIGHTS FOR NULL CHAR
            # if char1==self.null_char or char2==self.null_char:
                # continue
            self.trans_matrix[char1][char2] += 1
            self.totals[char1] += 1



    def update_params_all(self, pairs):
        '''Update all paramaters for a collection of pairs'''
        print("Updating model params")
        self.prev_trans_matrix = copy.deepcopy(self.trans_matrix)

        for pair in pairs:
            if pair[1] not in self.seen[pair[0]]:
                self.seen[pair[0]].add(pair[1])
                self.update_params(pair)


    def check_convergence(self):
        return self.trans_matrix == self.prev_trans_matrix



    def find_neg_log_prob(self, source, target):
        '''Find log probability of source --> target using trans matrix'''
        ops = self.augmented_ops(source, target)
        log_prob = 0
        for op in ops:
            char1, char2 = self.op2chars(op, source, target)
            try:
                log_prob += math.log(self.trans_matrix[char1][char2]/self.totals[char1])
            except:
                continue
#                 print(char1, char2)
#                 print(source, target)
#                 print(self.trans_matrix[char1][char2])
#                 print(self.totals[char1])
#                 print(self.trans_matrix[char1][char2]/self.totals[char1])



        return -log_prob


    # def get_sem_candidates(self, word, cand_target_words, K = 50):
    #     '''Get semantics-based candidates for a word based on bilingual embeddings'''

    #     if word not in self.sem_nns:
    #         nns = self.model.nearest_neighbors(word, k = K)
    #         self.sem_nns[word] = {nn for nn in nns if nn[0] in cand_target_words}

    #     if not self.training and word in cand_target_words:
    #         self.sem_nns[word].add((word,1))
    #         # print("2", word in self.sem_nns[word])

    #     return self.sem_nns[word]


    def find_best_match(self, source, cand_target_words, num_targets = 5):
        '''Find best match for source over cand words'''
        if self.training:
            min_dist, best_word = math.inf, ""
            for cand in cand_target_words:
                if cand.strip() == "" or not self.is_dev(cand):
                    continue
                dist_score = self.find_neg_log_prob(source, cand)
                if dist_score < min_dist:
                    min_dist = dist_score
                    best_word = cand
            return [(best_word, min_dist)]

        # cand_target_words.append(source)
        dist_scores = [(cand, self.find_neg_log_prob(source, cand)) \
            for cand in cand_target_words if self.is_dev(cand)]
        best_pairs = sorted(dist_scores, key = lambda x:x[1])[:num_targets]

        return best_pairs


    def dump_model_params(self):
        '''Save model params for inspection and continued training'''
        if not os.path.isdir(self.PARAMS_DIR):
            os.makedirs(self.PARAMS_DIR)

        with open(self.PARAMS_DIR+"trans_matrix.json", "w") as f:
            json.dump(self.trans_matrix, f, ensure_ascii = False, indent = 2)

        with open(self.PARAMS_DIR+"totals.json", "w") as f:
            json.dump(self.totals, f, ensure_ascii = False, indent = 2)


        with open(self.PARAMS_DIR+"seen.json", "w") as f:
            seen = {source:list(target_set) for source, target_set in self.seen.items()}
            json.dump(seen, f, ensure_ascii = False, indent = 2)


    def load_model_params(self):
        '''Initialize by loading pre-trained params'''

        
        with open(self.PARAMS_DIR+"trans_matrix.json", "r") as f:
            self.trans_matrix = defaultdict(lambda:defaultdict(lambda:0), json.load(f)) 

        with open(self.PARAMS_DIR+"totals.json", "r") as f:
            self.totals = defaultdict(lambda:1, json.load(f))


        with open(self.PARAMS_DIR+"seen.json", "r") as f:
            self.seen = json.load(f)
            self.seen = {source:set(target_set) for source, target_set in self.seen.items()}
            self.seen = defaultdict(lambda:set(), self.seen)





    def update(self, words_learnt):
        
        # word_pairs = [(elem["sent_info"]["word"], elem["top_cand"]) for elem in words_learnt]
        # threshold = min([elem["score"] for elem in words_learnt])
        if self.batch_no % 100 == 0 and self.batch_no:
            self.dump_model_params()

        # RECOMPUTE SCORES BASED ON NED TO FILTER BAD MATCHES
        for elem in words_learnt:
            elem["score"] = self.sim_func(elem["sent_info"]["word"], elem["top_cand"])

        updating_pairs, residue = super().update(words_learnt)

        self.update_params_all(updating_pairs)

        return updating_pairs, residue

    def main(self) -> None:
        '''Initialize model, tokenizer, approach variables'''

        # args = self.get_args()

        self.initializations()

        load_pretrained = False
        if os.path.isfile(self.PARAMS_DIR+"trans_matrix.json"):
            load_pretrained = True
        self.initialize_trans_matrix(load_pretrained = load_pretrained)

        atexit.register(self.save_lexicon)
        
        self.training = True
        self.driver()
        self.training = False
       

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for Rulebook BLI")
    parser.add_argument("--hf_model_name", type=str, required=True, \
        help="HF Model to use for pipelining MLM inference")
    parser.add_argument("--TARGET_FILE_PATH", type=str, required=True, \
        help="Filepath for data in target language/dialect")
    parser.add_argument("--batch_size", type=int, default=100, \
        help="Number of sentences processed together, after which bilingual lexicon is updated")
    parser.add_argument("--threshold", type=float, default=0.5, \
        help="Threshold for similarity function, if applicable")
    parser.add_argument("--OUTDIR", type=str, required=True, \
        help="Outdir for learned bilingual lexicon")
    parser.add_argument("--PARAMS_DIR", type=str, required=True, \
        help="Outdir for learned parameters")
    parser.add_argument("--iterations", type=int, default=1, \
        help="Number of iterations of the whole data (algorithm performs revisions on previously learnt words).")
    parser.add_argument("--lang", type=str, default="unk", \
        help="Target language")
    
    args = parser.parse_args()
    return args
    

if __name__=="__main__":

    args = get_args()
     
    # HF_MODEL_NAME="google/muril-base-cased"
    # TARGET_FILE_PATH = "../../data/monolingual/all/bho.txt"
    
    # batch_size = 100
    # iterations = 3
    # threshold = 0.5
    # sim_func_key = "ned"
    batch_size = args.batch_size
    iterations = args.iterations
    threshold = args.threshold
    lang = args.lang
    HF_MODEL_NAME = args.hf_model_name
    OUTDIR = args.OUTDIR
    PARAMS_DIR = args.PARAMS_DIR
    TARGET_FILE_PATH = args.TARGET_FILE_PATH

    sim_func_key = "ned"

    PARAMS_DIR = "{}/rulebook_noninit_allowid.simfunc_{}.batchsize_{}.iterations_{}.threshold_{}/".format(\
        PARAMS_DIR, sim_func_key, batch_size, iterations, threshold)
    if not os.path.isdir(PARAMS_DIR):
        os.makedirs(PARAMS_DIR)
    OUTPATH = "{}/rulebook_noninit_allowid.{}.simfunc_{}.batchsize_{}.iterations_{}.threshold_{}.json".format(\
        OUTDIR, lang, sim_func_key, batch_size, iterations, threshold)
    
    bli_obj = BLIRulebook(HF_MODEL_NAME=HF_MODEL_NAME, \
        TARGET_FILE_PATH=TARGET_FILE_PATH, OUTPATH=OUTPATH, PARAMS_DIR=PARAMS_DIR, \
        batch_size=batch_size, iterations=iterations, threshold=threshold)
    bli_obj.main()

# %%
