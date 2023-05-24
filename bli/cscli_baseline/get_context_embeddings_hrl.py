# We want to get contextual embeddings for the top most frequent words in the corpus
# First, we need to get the top 50000 words in the corpus
# Then, we randomly choose 15 contexts for each word
# Finally, we get the contextual embeddings for each word in those contexts, first averaged over subwords, and then over contexts
# We save the words and embeddings in a vec file format


import argparse
import numpy as np
import torch

from collections import defaultdict

from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os, sys
sys.path.append("../../")
from utils import get_tokenizers

# import atexit

def read_data(filename) -> str:
    '''Read data from a file line by line (for large files)'''
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()


def get_top_words(filename: str, num_words = 50000) -> list:
    '''Get the top most frequent words in the corpus'''
    print("Getting top ", num_words, " words from ", filename, " ...")
    word_counts = defaultdict(lambda: 0)
    for line in read_data(filename):
        for word in line.split():
            word_counts[word] += 1

    sorted_word_counts = sorted(word_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_words = [word for word, _ in sorted_word_counts[:num_words]]

    return top_words


def get_lm(tokenizer_path: str, model_path: str) -> tuple:
    '''Get the language model and tokenizer'''

    tokenizer = get_tokenizers.load_tokenizer(tokenizer_path)
    model = BertModel.from_pretrained(model_path)

    return tokenizer, model


def get_word_embeddings_from_model(tokenizer, model, sentences: list, word_indices: list) -> np.array:
    '''We want the word embedding of a word in a sentence, given the word index
    The index is based on a space-splitted list of tokens in the sentence.
    sentences: list of sentences, each sentence is a list of words
    word_indices: list, each element is a pair of (sentence_index, word_index)
    '''

    # Tokenize the sentence
    tokenized_sentence = tokenizer(sentences, truncation = True, max_length = 512, padding = True, is_split_into_words = True, return_tensors='pt')

    # Get the hidden states from the model
    last_hidden_state = model(**tokenized_sentence).last_hidden_state

    # print("Length of last_hidden_state: ", len(last_hidden_state))
    # print("Shape of last_hidden_state: ", last_hidden_state.shape)

    # Get the indices of the subwords corresponding to the word
    for sentence_index, word_index in word_indices:
        word_subword_indices = tokenized_sentence.word_to_tokens(batch_or_word_index = sentence_index, word_index = word_index)
        if word_subword_indices is None:
            yield None

        # Get the hidden states corresponding to the word
        try:
            word_embedding = np.mean(last_hidden_state[sentence_index, word_subword_indices[0]:word_subword_indices[-1], :].detach().numpy(), axis = 0)
        except:
            print("Error in getting word embedding!")
            print("Sentence index: ", sentence_index)
            print("Word index: ", word_index)
            print("Word subword indices: ", word_subword_indices)
            print("Sentence: ", sentences[sentence_index])
            print("Tokenized sentence: ", tokenized_sentence[sentence_index])
            print("Last hidden state: ", last_hidden_state)
            raise

        yield word_embedding
   
    # return word_hidden_states


def build_contextual_embeddings_matrix(filename: str, tokenizer, model, output_path, \
                                       top_words: list, num_contexts: int = 15, batch_size = 32) -> np.array:
    '''Build the contextual embeddings matrix for the top words in the corpus'''
    context_matrix = defaultdict(lambda: (None, 0)) # {word: (embedding, number of embeddings we've summed so far)}

    max_contexts = 15
    # atexit.register(save_embeddings, context_matrix, output_path)
    # atexit.register(print, len(context_matrix), " words processed so far.")

    sentences = list()
    processing_words = list()
    word_indices = list()

    words_to_be_found = len(top_words)
    sents_processed = 0

    for sentence in read_data(filename):
        sents_processed += 1
        words = sentence.split()

        sentences.append(words)

        at_least_one_word = False

        for idx, word in enumerate(words):          
            if word in top_words and context_matrix[word][1] < max_contexts:        
                at_least_one_word = True
                word_indices.append((len(sentences) - 1, idx))
                processing_words.append(word)
                # print(len(processing_words), " words in processing_words so far.", end = "\r")
                # print(len(word_indices), " word indices in processing so far.", end = "\r")
        
        if not at_least_one_word:
            sentences.pop()
            continue

        if len(sentences) == batch_size:
            # print(len(processing_words), " words in processing_words, before assert.", end = "\r")
            # print(len(word_indices), " word indices in processing before assert.", end = "\r")
            assert len(processing_words) == len(word_indices)
            assert len(processing_words) == len(all_embs)
            all_embs = list(get_word_embeddings_from_model(tokenizer, model, sentences, word_indices))
            
            # for word, embedding in zip(processing_words, get_word_embeddings_from_model(tokenizer, model, sentences, word_indices)):
            for word, embedding in zip(processing_words, all_embs):
                if embedding is not None:
                    if context_matrix[word][0] is None:
                        context_matrix[word] = (embedding, 1)
                    else:
                        context_matrix[word] = (context_matrix[word][0] + embedding, context_matrix[word][1] + 1)
                        if context_matrix[word][1] == max_contexts:
                            words_to_be_found -= 1
                else:
                    print("Word embedding is None!")
                    # print("Word: ", word)
                            
            sentences = []
            word_indices = []
            processing_words = []
        
        if len(context_matrix) % 1000 == 0:
            print("Number of words processed: ", len(context_matrix))
            print("Number of words to be found: ", words_to_be_found)
            print("Number of sentences processed: ", sents_processed)
            save_embeddings(context_matrix, output_path, num_contexts)
        
        if words_to_be_found == 0:
            print("All words found!")
            print("Number of sentences processed: ", sents_processed)
            break
    
    return context_matrix
    
    

def save_embeddings(context_matrix: dict, output_path: str, num_contexts: int = 15):

    # global required_words
    # Get the average embeddings
    context_matrix = {word: context_matrix[word][0] / context_matrix[word][1] \
                      for word in context_matrix \
                        if context_matrix[word][0] is not None \
                      and context_matrix[word][1] >= num_contexts}
    
    

    # context_matrix = {word: context_matrix[word][0] / context_matrix[word][1] \
    #                   for word in context_matrix \
    #                     if context_matrix[word][0] is not None}
    # Find 20000 most frequent words
    # top_words = sorted(context_matrix, key = lambda x: context_matrix[x][1], reverse = True)[:required_words]

    # min_freq = context_matrix[top_words[-1]][1]
    # print("Minimum frequency which we allowed: ", min_freq)

    # context_matrix = {word: context_matrix[word] for word in top_words}

    print("Number of words saved: ", len(context_matrix))
    
    # Save the embeddings
    with open(output_path, 'w') as f:
        f.write(str(len(context_matrix)) + ' ' + str(768) + '\n')
        for word, embedding in context_matrix.items():
            f.write(word + ' ' + ' '.join([str(x) for x in embedding]) + '\n')


def main(args):

    # args: corpus_path, tokenizer_path, model_path, output_path, num_words
    
    # global required_words
    # required_words = 20000
    min_contexts = 10

    top_words = get_top_words(args.corpus_path, args.num_words)

    tokenizer, model = get_lm(args.tokenizer_path, args.model_path)

    context_matrix = build_contextual_embeddings_matrix(args.corpus_path, tokenizer, model, args.output_path, \
                                                        top_words, num_contexts=min_contexts, batch_size = args.batch_size)

    save_embeddings(context_matrix, args.output_path, min_contexts)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, help='Path to the corpus')
    parser.add_argument('--tokenizer_path', type=str, help='Path to the tokenizer')
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--output_path', type=str, help='Path to the output file')
    parser.add_argument('--num_words', type=int, default=100000, help='Number of words to get embeddings for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    main(args)


