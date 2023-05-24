from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function
from gensim.models import FastText


from fasttext import load_model
import argparse
import errno
import fasttext
import fasttext.util
from scipy.spatial import distance
import re


def convert_bin2vec(fname, fout_file):
    
    f = load_model(fname)
    words = f.get_words()
    fout = open(fout_file, "w")
    fout.write(str(len(words)) + " " + str(f.get_dimension())+"\n")

    error = 0
    for w in words:
        v = f.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            fout.write(w + vstr+ "\n")
        except IOError as e:
            error += 1
            if e.errno == errno.EPIPE:
                pass

# %%
# MONOLINGUAL DATAPATH
dims = 100
# langs = {"bho", "hin"}
# langs = {"mag", "bho"}
langs = {"hin"}
# langs = {"mar", "nep"}
for lang in langs:
    print(lang)
    DATAPATH = "../../data/monolingual/all/{}.txt".format(lang)
    EMBS_PATH = "embeddings/monolingual/{}.dims_{}.bin".format(lang, str(dims))
    EMBS_PATH_VEC = "embeddings/monolingual/{}.dims_{}.vec".format(lang, str(dims))
    # %%
    # TRAINING
    model = fasttext.train_unsupervised(DATAPATH, dim = dims)
    print("Trained!")
    # %%
    # SAVING MODEL
    model.save_model(EMBS_PATH)
    print("Saved in .bin format!")
    convert_bin2vec(EMBS_PATH, EMBS_PATH_VEC)
    print("Saved in .vec format!")
