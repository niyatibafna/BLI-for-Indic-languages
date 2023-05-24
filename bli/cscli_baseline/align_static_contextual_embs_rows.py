# Given the static and contextual embeddings, we want to align them such that
# the same row in the two files corresponds to the same word.
# We will then write the aligned embeddings to a vec file.

import numpy as np
import io
import os, sys

def get_filepaths(lang):
    '''Get filepaths for static and contextual embeddings'''
    c_embs_filepath_dict = {
        "bho":"contextual_embeddings/bho/lm_(lm_mono.bho.batchsize_16.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec",
        "mag":"contextual_embeddings/mag/lm_(lm_mono.mag.batchsize_64.vocabsize_30522.epochs_40).min,maxcontexts_2,10.vec",
        "hin":"contextual_embeddings/hin/lm_pt_.min,maxcontexts_5,15.vec",
        "mar":"contextual_embeddings/mar/lm_pt_.min,maxcontexts_5,15.vec",
        "nep":"contextual_embeddings/nep/lm_pt_.min,maxcontexts_5,15.vec",
    }
    s_embs_filepath = "../bli_with_vecmap/embeddings/mapped/{}.dims_300.vec".format(lang)
    c_embs_filepath = c_embs_filepath_dict[lang]
    s_embs_out_filepath = "embeddings_row_aligned/static/{}.dims_300.vec".format(lang)
    c_embs_out_filepath = "embeddings_row_aligned/contextual/{}.dims_768.vec".format(lang)

    os.makedirs("embeddings_row_aligned/static", exist_ok=True)
    os.makedirs("embeddings_row_aligned/contextual", exist_ok=True)

    return s_embs_filepath, c_embs_filepath, s_embs_out_filepath, c_embs_out_filepath

def read_embs(filepath):
    embs = list()
    with open(filepath, "r") as f:
        for line in f:
            embs.append(line.strip().split())
    return embs

def read_txt_embeddings(emb_path, size=-1):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []
    max_size = 100000000 if size == -1 else size
    # load pretrained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                assert len(split) == 2
            else:
                word, vect = line.rstrip().split(' ', 1)
                word = word.lower()
                vect = np.fromstring(vect, sep=' ', dtype='float')
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    print('word have existed')
                else:
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            if i == size:
                break

    assert len(word2id) == len(vectors)
    # compute new vocabulary / embeddings
    id2word = {v: k for k, v in word2id.items()}

    embeddings = np.concatenate(vectors, 0)

    return word2id, id2word, embeddings 

def write_embs(word2id, embs, filepath):
    '''Write embedding in vec format to file'''

    with open(filepath, "w") as f:
        f.write(str(len(word2id)) + " " + str(len(embs[0])) + "\n")
        for word in word2id:
            emb = embs[word2id[word]]
            # Convert numpy emb to str and write
            emb = [str(x) for x in emb]
            f.write(word + " " + " ".join(emb) + "\n")    

def align_embs(s_word2id, c_word2id, s_embs, c_embs):

    s_word2id_aligned = dict()
    c_word2id_aligned = dict()
    s_embs_aligned = list()
    c_embs_aligned = list()

    for word in s_word2id:
        if word in c_word2id:
            s_word2id_aligned[word] = len(s_word2id_aligned)
            c_word2id_aligned[word] = len(c_word2id_aligned)
            s_embs_aligned.append(s_embs[s_word2id[word]])
            c_embs_aligned.append(c_embs[c_word2id[word]])

    assert len(s_word2id_aligned) == len(c_word2id_aligned) == len(s_embs_aligned) == len(c_embs_aligned)
    print("Number of aligned words: ", len(s_word2id_aligned))

    return s_word2id_aligned, c_word2id_aligned, s_embs_aligned, c_embs_aligned

def main():

    LANGS = ["mar", "nep"]
    for lang in LANGS:
        print("Aligning embeddings for {}...".format(lang))
        # Get filepaths
        s_embs_filepath, c_embs_filepath, s_embs_out_filepath, c_embs_out_filepath = get_filepaths(lang)
        # Read embeddings
        s_word2id, s_id2word, s_embs = read_txt_embeddings(s_embs_filepath)
        c_word2id, c_id2word, c_embs = read_txt_embeddings(c_embs_filepath)
        # Align embeddings
        s_word2id_aligned, c_word2id_aligned, s_embs_aligned, c_embs_aligned = align_embs(s_word2id, c_word2id, s_embs, c_embs)
        # Write embeddings
        write_embs(s_word2id_aligned, s_embs_aligned, s_embs_out_filepath)
        write_embs(c_word2id_aligned, c_embs_aligned, c_embs_out_filepath)
    
if __name__ == "__main__":
    main()

