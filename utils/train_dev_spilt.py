from sklearn.model_selection import train_test_split
from variables import LANGS as LANGS
import os

SEED = 42

DATADIR = "../data/monolingual/equal_subsets_tokens/"
OUTDIR = "../data/monolingual/training_splits_eqsubtok/"
TRAIN_DIR = OUTDIR + "train/"
DEV_DIR = OUTDIR + "dev/"

if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)

if not os.path.isdir(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)

if not os.path.isdir(DEV_DIR):
    os.mkdir(DEV_DIR)


for lang in LANGS:
    # if lang == "awa":
        # continue
    FILEPATH = DATADIR + lang + ".txt"
    TRAIN_OUT = TRAIN_DIR + lang + ".txt"
    DEV_OUT = DEV_DIR + lang + ".txt"
    with open(FILEPATH, "r", encoding="utf-8") as in_f,\
        open(TRAIN_OUT, "w", encoding="utf-8") as out_train,\
            open(DEV_OUT, "w", encoding="utf-8") as out_dev:
        
        lines = in_f.read().split("\n")
        train_data, dev_data = train_test_split(lines, test_size = 0.1, random_state = SEED)

        for line in train_data:
            out_train.write(line+"\n")
        
        for line in dev_data:
            out_dev.write(line+"\n")


