# %%

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM, BertForMaskedLM
from transformers import pipeline, FillMaskPipeline
from collections import defaultdict
import editdistance
import math, random
import json
import os, sys
import argparse
import atexit
sys.path.append("../../")
from utils import get_tokenizers
from tqdm import tqdm
import torch
# %%

# Subclass the FillMaskPipeline to truncate inputs when tokenizing
class FillMaskPipelineTrunc(FillMaskPipeline):

    def preprocess(self, inputs, return_tensors=None, **preprocess_parameters):
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, truncation=True, max_length=512, return_tensors=return_tensors)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs

# def preprocess_truncate(self, inputs, return_tensors=None, **preprocess_parameters):
#     if return_tensors is None:
#         return_tensors = self.framework
#     model_inputs = self.tokenizer(inputs, truncation=True, max_length=512, return_tensors=return_tensors)
#     self.ensure_exactly_one_mask_token(model_inputs)
#     return model_inputs


class PipeInput(torch.utils.data.Dataset):

        def __init__(self, data) -> None:
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]["input"]
        def get_sent_info(self, idx):
            return self.data[idx]


class BLIBasic:

    def __init__(self, TOKENIZER_INPATH = None, MODEL_INPATH = None, HF_MODEL_NAME = None,\
        TARGET_FILE_PATH = None,\
        OUTPATH = "lexicon.json",\
        batch_size = 500, iterations = 3, threshold = 0.5) -> None:
        self.mask_token = "[MASK]"
        self.TOKENIZER_INPATH = TOKENIZER_INPATH
        self.MODEL_INPATH = MODEL_INPATH
        self.HF_MODEL_NAME = HF_MODEL_NAME
        self.TARGET_FILE_PATH = TARGET_FILE_PATH
        self.OUTPATH = OUTPATH
        self.SOURCE_PATH = ""
        if os.path.isfile(self.OUTPATH):
            self.SOURCE_PATH = OUTPATH

        self.bil_dict = defaultdict(lambda : dict())

        self.batch_size = batch_size
        self.min_iterations = iterations
        self.batch_no = 0


        ned_sim = lambda a, b : \
            1 - editdistance.eval(a, b)/max(len(a), len(b))
        # jwm = lambda a, b : \
        # jaro.jaro_winkler_metric(a, b)
        self.sim_func = ned_sim        
        self.threshold = threshold
        self.max_length = 512

    def is_dev(self, s):
        dev_range = range(2305, 2404)
        for c in s:
            if ord(c) not in dev_range:
                return False
        return True
    
    def get_model_and_tokenizer(self) -> None:
        '''Initialize model pipeline for masked LM, and tokenizer vocabulary'''
        if not self.HF_MODEL_NAME:
            assert os.path.isfile(self.TOKENIZER_INPATH)
            assert os.path.isdir(self.MODEL_INPATH)
            # logging.info("Loading from path! ")
            print("Loading pretrained!")
            self.tokenizer = get_tokenizers.train_or_load_tokenizer(self.TOKENIZER_INPATH)
            # self.pipe = pipeline("fill-mask", model = self.MODEL_INPATH, \
            # tokenizer = self.tokenizer, top_k = 30)

            # TODO: Fix this
            self.pipe = FillMaskPipelineTrunc(task = "fill-mask", model = self.MODEL_INPATH, \
            tokenizer = self.tokenizer, top_k = 30)
            # self.model = BertForMaskedLM.from_pretrained(self.MODEL_INPATH)
        else:
            # logging.info("Loading from HF! ")
            print("Loading HF {}".format(self.HF_MODEL_NAME))
            self.tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_NAME, model_max_length=512)
            # self.pipe = pipeline("fill-mask", model = self.HF_MODEL_NAME, \
            # tokenizer = self.tokenizer, top_k = 30)
            # self.pipe.preprocess = preprocess_truncate
            model = AutoModelForMaskedLM.from_pretrained(self.HF_MODEL_NAME)
            self.pipe = FillMaskPipelineTrunc(task = "fill-mask", model = model, \
            tokenizer = self.tokenizer, top_k = 30)
            # self.model = AutoModelForMaskedLM.from_pretrained(self.HF_MODEL_NAME)


    def get_target_data(self):
        '''Read target data file'''
        with open(self.TARGET_FILE_PATH, "r", encoding="utf8") as f:
            data = f.read().split("\n")

        # with open("sample.txt", "r", encoding="utf8") as f:
        #     data = f.read().split("\n")

        self.rem_sentences = [{"text": sent, "processed": 0} for sent in data]

    def get_bilingual_dictionary(self):
        '''Initialize known source vocabulary'''
        if self.SOURCE_PATH:
            print("Loading precomputed!")
            with open(self.SOURCE_PATH, "r") as f:
                precomputed = json.load(f)
            self.bil_dict = defaultdict(lambda : dict(), precomputed)
            if "batch_no" in self.bil_dict:
                self.batch_no = self.bil_dict["batch_no"]
            else:
                self.batch_no = 1
        else:
            print("Initializing as empty!")
        # else:
        #     print("Initializing from tokenizer vocabulary!")
        #     tokenizer_vocab = set(self.tokenizer.vocab.keys())
        #     # print("Length of source vocab: {}".format(len(source_vocab.keys())))
        #     for sent in self.rem_sentences:
        #         words = sent["text"].split()
        #         for word in words:
        #             if word in tokenizer_vocab:
        #                 self.bil_dict[word][word] = 1


    def post_init(self):
        '''Add identical entries when word is present in the Hindi tokenizer'''
        self.get_target_data()
        tokenizer_vocab = set(self.tokenizer.vocab.keys())

        for sent in self.rem_sentences:
            words = sent["text"].split()
            for word in words:
                if word in tokenizer_vocab:
                    self.bil_dict[word][word] = 1


    
    def order_rem_sentences(self):

        new_rem_sentences = list()

        for idx, sent in enumerate(self.rem_sentences):
            words = sent["text"].split()
            known, unknown  = 0, 0
            for word in words:
                source_rep = self.get_trans(word)
                if source_rep:
                    known += 1
                else:
                    unknown += 1

            if unknown != 0:
                self.rem_sentences[idx]["uncertainty"] = unknown/len(words)
                new_rem_sentences.append(self.rem_sentences[idx])

        self.rem_sentences = [x for x in new_rem_sentences]

        self.rem_sentences = sorted(self.rem_sentences, key=lambda x: (x["processed"], x["uncertainty"]))

    


    def sentence_handler(self, residue):

        for idx in range(len(residue)):
            residue[idx]["processed"] += 1
        self.rem_sentences += residue

        #Optional
        self.order_rem_sentences()

        # This is the number of iterations that each sentence has definitely had
        curr_iterations = self.rem_sentences[0]["processed"]

        batch_sents = self.rem_sentences[:self.batch_size]
        self.rem_sentences = self.rem_sentences[self.batch_size:]

        batch = list()
        for sent in batch_sents:
            # Create input to pipe
            words = sent["text"].split()[:self.max_length]

            # Replace all known words with their equivalents, keep track of unknown words
            unknown_set = set()
            for idx, word in enumerate(words):
                source_rep = self.get_trans(word)
                if source_rep:
                    words[idx] = source_rep
                else:
                    if self.is_dev(word):
                        unknown_set.add(word)

            # Create different batch element for each unknown word
            for word in unknown_set:
                # Mask all instances of the word in the sentence
                masked = [w if w!=word else self.mask_token for w in words]
                masked = " ".join(masked)
                element = {"text":sent["text"], \
                            "input": masked, \
                            "word": word,\
                            "processed": sent["processed"]}
                batch.append({k:v for k,v in element.items()})
            
        # batch = PipeInput(batch)

        return batch, curr_iterations


    def terminate_learning(self, curr_iterations):
        if curr_iterations >= self.min_iterations:
            return True
        return False


    def get_trans(self, word) -> str:
        if word not in self.bil_dict:
            return ""
        cands_probs = list(self.bil_dict[word].items())
        cands = [p[0] for p in cands_probs]
        probs = [p[1] for p in cands_probs]
        sample_cand = random.choices(cands, weights=probs)[0]

        return sample_cand


    def find_best_match(self, word, cand_target_words, num_targets = 1):
        '''Find best match using NED'''
        # vowel_range = list(range(2305, 2315)) + list(range(2317, 2325)) + list(range(2365, 2384))
        # bad_char_range = range(2364, 2367)

        sim_scores = [(cand, self.sim_func(word, cand)) for cand in cand_target_words if self.is_dev(cand)]

        best_pairs = sorted(sim_scores, key = lambda x:x[1], reverse=True)[:num_targets]
        if best_pairs:
            return best_pairs


    def train(self, batch):
        '''
        Returns: {"sent_info":sent, "top_cand": top_cand, "score": score},
        where sent is same as elem of self.rem_sentences'''

        batch_sents = [sent["input"] for sent in batch]
        try:
            output = self.pipe(batch_sents, batch_size = min(self.batch_size, 100))
            print("Output obtained!", flush=True)
        except:
            print("Skipped batch because no [MASK] token found!")
            return list()
            

        words_learnt = list()
        for idx, results in enumerate(tqdm(output)):
            # sent = batch.get_sent_info(idx)
            sent = batch[idx]
            # mlm_input = sent["input"] 
            # results = self.pipe(mlm_input)
            try:
                if type(results[0]) == list:
                    candidates = [each_word["token_str"] for each_mask_output in results for each_word in each_mask_output]
                    weights = [each_word["score"] for each_mask_output in results for each_word in each_mask_output]
                else:
                    candidates = [result["token_str"] for result in results]
                    weights = [result["score"] for result in results]
            except:
                print(results)
                pass
            

            best_pairs = self.find_best_match(sent["word"], candidates, num_targets = 1)
            if best_pairs:
                top_cand, score = best_pairs[0]
                if top_cand and score:
                    words_learnt.append({"sent_info":sent, "top_cand": top_cand, "score": score})


        return words_learnt


    def save_lexicon(self):
        '''Stores lexicon in specified location'''

        # number of sentences processed = self.batch_no * self.batch_size
        # The above FAILS if you change the batch size in the meantime. 
        # I could handle this problem without too much difficulty but honestly have better things to do.
        self.bil_dict["batch_no"] = self.batch_no
        self.bil_dict["batch_size"] = self.batch_size

        # Add identical words to the lexicon
        self.post_init()

        with open(self.OUTPATH, "w") as f:
            json.dump(self.bil_dict, f, indent=2, ensure_ascii = False)
    

    def update(self, words_learnt, threshold = None):

        
        if not threshold:
            threshold = self.threshold

        if self.batch_no % 10 == 0 and self.bil_dict:
            self.save_lexicon()
        
        residue = list()
        residue_sents = set()

        updating_pairs = list()
        for elem in words_learnt:

            word, cand, score, sent = elem["sent_info"]["word"], \
                elem["top_cand"], elem["score"], elem["sent_info"]
            if score >= threshold:
                self.bil_dict[word][cand] = score
                updating_pairs.append((word, cand))
            else:
                if sent["text"] not in residue_sents:
                    residue.append(sent)
                    residue_sents.add(sent["text"])

        return updating_pairs, residue
    
    def initializations(self):
        '''Initialize model, tokenizer, approach variables'''
        print("Initializing...")
        self.get_model_and_tokenizer()
        print("Model and tokenizer loaded!")
        self.get_target_data()
        print("Total sentences: {}".format(len(self.rem_sentences)))
        self.get_bilingual_dictionary()
        print("Initial length of bilingual dictionary: {}".format(len(self.bil_dict)))
        
    def driver(self):
        residue = list()
        # Save lexicon every time 1000 new words are learnt
        curr_target = 1000
        while True:
            batch, curr_iterations = self.sentence_handler(residue)
            if self.terminate_learning(curr_iterations):
                break
            self.batch_no += 1
            print("Number of iterations completed: {}".format(curr_iterations))
            print("Processing batch {} of size: {}".format(self.batch_no, len(batch)))
            candidate_pairs = self.train(batch)
            updating_pairs, residue = self.update(candidate_pairs)
            print("WORDS LEARNT: {}".format(len(updating_pairs)))
            print("Examples: {}".format(str(updating_pairs[:5])))
            print("Length of bilingual dictionary: {}".format(len(self.bil_dict)))
            if len(self.bil_dict) >= curr_target:
                self.save_lexicon()
                curr_target = len(self.bil_dict) + 1000



    def main(self) -> None:
        print("Starting learning process...")
        self.initializations()
        atexit.register(self.save_lexicon)
        self.driver()
        

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for Basic BLI")
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
    parser.add_argument("--iterations", type=int, default=1, \
        help="Number of iterations of the whole data (algorithm performs revisions on previously learnt words).")
    parser.add_argument("--lang", type=str, default="unk", \
        help="Target language")
    
    args = parser.parse_args()
    return args


if __name__=="__main__":

    print("Running Basic BLI")
    args = get_args()

    batch_size = args.batch_size
    iterations = args.iterations
    threshold = args.threshold
    lang = args.lang
    HF_MODEL_NAME = args.hf_model_name
    OUTDIR = args.OUTDIR
    TARGET_FILE_PATH = args.TARGET_FILE_PATH

    sim_func_key = "ned"
    OUTPATH = "{}/basic_noninit+hintokvocab.{}.simfunc_{}.batchsize_{}.iterations_{}.threshold_{}.json".format(\
        OUTDIR, lang, sim_func_key, batch_size, iterations, threshold)
    # HF_MODEL_NAME="google/muril-base-cased"
    # TARGET_FILE_PATH = "../../data/monolingual/all/bho.txt"
    # batch_size = 100
    # iterations = 3
    # threshold = 0.5
    # lang = "bho"
    
    # OUTDIR = "../lexicons/bho"    
    print(OUTPATH)
    
    bli_obj = BLIBasic(HF_MODEL_NAME=HF_MODEL_NAME, \
        TARGET_FILE_PATH=TARGET_FILE_PATH, OUTPATH=OUTPATH, \
        batch_size=batch_size, iterations=iterations, threshold=threshold)

    bli_obj.main()
