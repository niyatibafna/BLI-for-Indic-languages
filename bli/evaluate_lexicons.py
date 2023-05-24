# Get gold lexicon
# Get target lexicon
# Find intersection - this is the total word set we will evaluate on ;
    # it's not fair to evaluate on words that were not in the data. 
    # We need a list of all words that occurred in the data (intersecting with gold lex) - this 
    # should be our eval set. Filter by frequency (remove 1 occurrence words)
# Of word set, for each word, find top k predictions by lexicon (k=2)
# If any of those are in the gold lexicon, count +1 
# We can calculate two measures: accuracy and coverage
# accuracy = correct/len(pred_lexicon \intersection gold_lexicon)
# coverage = correct/len(word_set)

# ACCURACY : Of the words that it claims it knows and can be evaluated, how many does it know?
# COVERAGE : Of all possible words that it's met and can be evaluated, how many does it know?

# %%
import json
from collections import Counter, defaultdict
import math

class EvaluationLexicons:

    def __init__(self, TARGET_FILE_PATH, GOLD_LEXICON_PATH, PRED_LEXICON_PATH,\
        top_k = 2, filter_freq_less_than = 2) -> None:
        self.TARGET_FILE_PATH = TARGET_FILE_PATH
        self.GOLD_LEXICON_PATH = GOLD_LEXICON_PATH
        self.PRED_LEXICON_PATH = PRED_LEXICON_PATH
        self.top_k = top_k
        self.filter_freq_less_than = filter_freq_less_than
        self.dataset_words = dict()

    
    def get_word_set(self):

        with open(self.TARGET_FILE_PATH, "r") as f:
            data = f.read().split()

        freq_list = Counter(data)
        
        self.dataset_words = defaultdict(lambda:0, \
            {k:v for k,v in freq_list.items() if v >= self.filter_freq_less_than})
        

    def lexicons(self, LEXICON_PATH):

        with open(LEXICON_PATH, "r") as f:
            lex = json.load(f)
        
        special_keys = {"batch_size", "batch_no"}
        for k in special_keys:
            if k in lex:
                del lex[k]
        
        return lex
        
    def evaluate(self):

        nonid_correct = set()
        nonid_wrong = set()
        identical_correct = set()
        identical_wrong = set()

        correct = set()
        wrong = set()

        nonid_prec = 0
        nonid_prec_count = 0

        total_eval_set = set(self.dataset_words.keys()).intersection(set(self.gold_lex.keys()))
        predicted_set = set(self.pred_lex.keys()).intersection(set(self.gold_lex.keys()))

        for word in predicted_set:

            words_scores = sorted(self.pred_lex[word].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            top_preds = {w[0] for w in words_scores}

            # CHOOSE WHETHER TO TAKE top_k OF GOLD LEXICON
            global hrl
            if hrl:
                gold_words_scores = sorted(self.gold_lex[word].items(), key=lambda x: x[1], reverse=True)
            else:
                gold_words_scores = sorted(self.gold_lex[word].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            gold_preds = {w[0] for w in gold_words_scores}

            common = top_preds.intersection(gold_preds)

            for p in top_preds:
                if p in common:
                    print(word, p)
                    if p == word:
                        identical_correct.add((word, p))
                    else:
                        nonid_correct.add((word, p))
                else:
                    if p == word:
                        identical_wrong.add((word, p))
                    else:
                        nonid_wrong.add((word, p))
            
            if len(common) > 0:
                correct.add(word)
            else:
                wrong.add(word)

            # Precision@2 for nonID preds
            for c in common:
                if c != word:
                    nonid_prec += 1
                    break
            
            has_id = False
            for g in gold_preds:
                if g == word:
                    has_id = True

            if not has_id:
                nonid_prec_count += 1
            

            


            # if len(common) > 0:
            #     # We have correct predictions. If all our predictions are identical, we put them in the identical_correct set
            #     all_identical = True
            #     for c in common:
            #         if word != c:
            #             all_identical = False
                
            #     if all_identical:
            #         identical_correct.add((word, c))
            #     else:
            #         correct.add((word, c))                        
            # else:
            #     all_identical = True
            #     for c in top_preds:
            #         if word != c:
            #             all_identical = False
                
            #     if all_identical:
            #         identical_wrong.add((word, c))
            #     else:
            #         wrong.add((word, c))


        print("NONID PRECISION@2: {}".format(nonid_prec/nonid_prec_count if nonid_prec_count > 0 else 0))
        print("nonid numbers: ", nonid_prec, nonid_prec_count)
        return correct, wrong, identical_correct, identical_wrong, nonid_correct, nonid_wrong,  total_eval_set, predicted_set


    def evaluate_assumeid(self):

        nonid_correct = set()
        nonid_wrong = set()
        identical_correct = set()
        identical_wrong = set()

        correct = set()
        wrong = set()

        # for word in set(self.gold_lex.keys()):
        #     if word not in self.pred_lex:
        #         self.pred_lex[word] = {word:1}

        # total_eval_set = set(self.dataset_words.keys()).intersection(set(self.gold_lex.keys()))
        total_eval_set = set(self.gold_lex.keys())
        predicted_set = set(self.pred_lex.keys()).intersection(set(self.gold_lex.keys()))

        for word in total_eval_set:
            if word in self.pred_lex:
                self.pred_lex[word][word] = math.inf
            else:
                # continue
                self.pred_lex[word] = {word:math.inf}

            words_scores = sorted(self.pred_lex[word].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            top_preds = {w[0] for w in words_scores}

            # CHOOSE WHETHER TO TAKE top_k OF GOLD LEXICON
            gold_words_scores = sorted(self.gold_lex[word].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            gold_preds = {w[0] for w in gold_words_scores}

            common = top_preds.intersection(gold_preds)
            for p in top_preds:
                if p in common:
                    if p == word:
                        identical_correct.add((word, p))
                    else:
                        nonid_correct.add((word, p))
                else:
                    if p == word:
                        identical_wrong.add((word, p))
                    else:
                        nonid_wrong.add((word, p))
            
            if len(common) > 0:
                correct.add(word)
            else:
                wrong.add(word)

        predicted_set = set(self.pred_lex.keys()).intersection(set(self.gold_lex.keys()))
        return correct, wrong, identical_correct, identical_wrong, nonid_correct, nonid_wrong,  total_eval_set, predicted_set

    def evaluate_allid(self):

        nonid_correct = set()
        nonid_wrong = set()
        identical_correct = set()
        identical_wrong = set()

        correct = set()
        wrong = set()
        # for word in set(self.gold_lex.keys()):
        #     if word not in self.pred_lex:
        #         self.pred_lex[word] = {word:1}

        total_eval_set = set(self.dataset_words.keys()).intersection(set(self.gold_lex.keys()))
        predicted_set = set(self.pred_lex.keys()).intersection(set(self.gold_lex.keys()))

        for word in self.gold_lex:
            
            self.pred_lex[word] = {word:1}

            words_scores = sorted(self.pred_lex[word].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            top_preds = {w[0] for w in words_scores}

            # CHOOSE WHETHER TO TAKE top_k OF GOLD LEXICON
            gold_words_scores = sorted(self.gold_lex[word].items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            gold_preds = {w[0] for w in gold_words_scores}

            common = top_preds.intersection(gold_preds)
            for p in top_preds:
                if p in common:
                    if p == word:
                        identical_correct.add((word, p))
                    else:
                        nonid_correct.add((word, p))
                else:
                    if p == word:
                        identical_wrong.add((word, p))
                    else:
                        nonid_wrong.add((word, p))
            
            if len(common) > 0:
                correct.add(word)
            else:
                wrong.add(word)

        total_eval_set = set(self.gold_lex.keys())
        predicted_set = set(self.pred_lex.keys()).intersection(set(self.gold_lex.keys()))
        return correct, wrong, identical_correct, identical_wrong, nonid_correct, nonid_wrong,  total_eval_set, predicted_set

    
    def find_eval_metrics(self):

        
        correct, wrong, identical_correct, identical_wrong,  nonid_correct, nonid_wrong, total_eval_set, predicted_set = \
            self.evaluate()
        # correct, wrong, identical_correct, identical_wrong,  nonid_correct, nonid_wrong, total_eval_set, predicted_set = \
        #     self.evaluate_assumeid()
        # correct, wrong, identical_correct, identical_wrong,  nonid_correct, nonid_wrong, total_eval_set, predicted_set = \
        #     self.evaluate_allid()

        total = len(correct) + len(wrong)
        accuracy = len(correct)/total
        print("Accuracy: {}".format(accuracy))
        coverage = 0
        # coverage = len(correct)/len(total_eval_set)
        # print("Coverage: {}".format(coverage))

        # accuracy = (len(identical_correct)+len(nonid_correct))/(len(identical_correct)+len(nonid_correct)+len(identical_wrong)+len(nonid_wrong))
        # print("Accuracy of all predictions (mult per word): {}".format(accuracy))

        total_identical = len(identical_correct)+len(identical_wrong)
        id_accuracy = len(identical_correct)/total_identical if total_identical > 0 else 0

        total_nonid = len(nonid_correct) + len(nonid_wrong)
        nonid_accuracy = len(nonid_correct)/total_nonid if total_nonid > 0 else 0

        print("Accuracy: {}".format(accuracy))
        print("Accuracy of identical predictions: {}".format(id_accuracy))
        print("Accuracy of non-identical predictions: {}".format(nonid_accuracy))
        print("Coverage: {}".format(coverage))
        


        print("All the following numbers relate to words in intersection with eval set")
        print("PREDICTED: {}".format(len(predicted_set)))
        # print("UNK WORDS IN DATASET (AND EVAL SET): {}".format(len(total_eval_set)))
        print("TOTAL IDENTICAL PREDICTED: {}".format(total_identical))
        print("Identical predictions: {}".format(total_identical/(total_identical + total_nonid)))

        print("TOTAL NON IDENTICAL PREDICTED: {}".format(total_nonid))
        print("TOTAL NON IDENTICAL CORRECT: {}".format(len(nonid_correct)))
        print("Identical predictions: {}".format(total_nonid/(total_identical + total_nonid)))

        print("CORRECTLY PREDICTED: {}".format(len(correct)))
        print("Identical correct instances: {}".format(len(identical_correct)))
        print("Non-identical correct instances: {}".format(len(nonid_correct)))
        print("Fraction of identical correct of all correct: {}".format(len(identical_correct)/len(correct) if len(correct) > 0 else 0))

        print("WRONGLY PREDICTED: {}".format(len(nonid_wrong)+ len(identical_wrong)))
        print("Identical wrong instances: {}".format(len(identical_wrong)))
        print("Non-identical wrong instances: {}".format(len(nonid_wrong)))


        print("Stats of predicted lexicon regardless of eval set")
        
        print("TOTAL PREDICTED: {}".format(len(self.pred_lex)))
        # print("TOTAL UNK WORDS IN DATASET: {}".format(len(self.dataset_words)))
        # print("% where some claim was made {}".format())

        with open("misc/non_id_lexicons/noninit_nonid_correct.txt", "w") as f:
            for w in nonid_correct:
                f.write(str(w)+"\n")
        with open("misc/non_id_lexicons/noninit_nonid_wrong.txt", "w") as f:
            for w in nonid_wrong:
                f.write(str(w)+"\n")

        print(len(predicted_set))
        print(round(accuracy*100,2))
        print(round(total_identical*100/(total_identical+total_nonid),2))
        print(round(id_accuracy*100,2))
        print(round(nonid_accuracy*100,2))
        print(round(len(identical_correct)*100/(len(identical_correct)+len(nonid_correct)),2) if len(identical_correct)+len(nonid_correct) > 0 else 0)
        print(len(self.pred_lex))
        print(len(self.dataset_words))
        
        print("\n\n\n")


        
        return accuracy, coverage
        
        

    def main(self):
        # self.get_word_set()
        self.gold_lex = self.lexicons(self.GOLD_LEXICON_PATH)
        self.pred_lex = self.lexicons(self.PRED_LEXICON_PATH)

        print("LENGTH OF EVAL LEXICON: {}".format(len(self.gold_lex)))
        print("LENGTH OF PRED LEXICON: {}".format(len(self.pred_lex)))
        # Number of entries with key value identical
        id = 0
        for k,v in self.pred_lex.items():
            if k in v:
                id+=1
        print("NUMBER OF IDENTICAL ENTRIES: {}".format(id))

        accuracy, coverage = self.find_eval_metrics()

        print("ACCURACY: {}".format(accuracy))
        print("COVERAGE: {}".format(coverage))




if __name__=="__main__":
    print("Starting!")
    global hrl
    hrl = True
    # TARGET_FILE_PATH = "../data/monolingual/all/bho.txt"
    # GOLD_LEXICON_PATH = "get_eval_lexicon/lexicons/target2hin/bho2hin.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/CSCBLI_unsup.bho.ft_300.ct_768.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/rulebook.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/basicnoned_noninit+hintokvocab.bho.simfunc_none.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "bli_with_vecmap/lexicons/bho.ft_300.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/bho.ft_100.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/basic.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/basic_noninit.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/rulebook_noninit.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/combined_basic+noninit.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/rulebook_noninit_allowid.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/basic_noninit+hintokvocab.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/bho/rulebook_noninit+hintokvocab.bho.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"

    # TARGET_FILE_PATH = "../data/monolingual/all/mag.txt"
    # GOLD_LEXICON_PATH = "get_eval_lexicon/lexicons/target2hin/mag2hin.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/CSCBLI_unsup.mag.ft_300.ct_768.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/rulebook_noninit.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/basic_noninit.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "bli_with_vecmap/lexicons/mag.ft_300.csls.json"
    # PRED_LEXICON_PATH = "bli_with_vecmap/lexicons/mag.ft_100.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/rulebook.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/rulebook_optpath_noupdatenull.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/basic.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/combined_basic+noninit.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/basic_noninit+hintokvocab.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/rulebook_noninit_allowid.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mag/rulebook_noninit+hintokvocab.mag.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"

    # TARGET_FILE_PATH = "../data/monolingual/all/mar.txt"
    # GOLD_LEXICON_PATH = "get_eval_lexicon/lexicons/target2hin/mar2hin.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mar/basic_noninit.mar.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mar/mar.ft_300.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/mar/CSCBLI_unsup.mar.ft_300.ct_768.csls.json"

    # TARGET_FILE_PATH = "../data/monolingual/all/nep.txt"
    # GOLD_LEXICON_PATH = "get_eval_lexicon/lexicons/target2hin/nep2hin.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/nep/rulebook_noninit_allowid.nep.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/nep/basic_noninit.nep.simfunc_ned.batchsize_100.iterations_3.threshold_0.5.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/nep/nep.ft_300.csls.json"
    # PRED_LEXICON_PATH = "../bli/lexicons/nep/CSCBLI_unsup.nep.ft_300.ct_768.csls.json"


    top_k = 2
    filter_freq_less_than = 2

    eval_obj = EvaluationLexicons(TARGET_FILE_PATH, GOLD_LEXICON_PATH, PRED_LEXICON_PATH,\
        top_k = top_k, filter_freq_less_than = filter_freq_less_than)

    eval_obj.main()
    




# %%
