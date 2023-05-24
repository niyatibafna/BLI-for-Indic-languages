from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import os
import gzip
from tqdm import tqdm

from datareader import MONO_DATADIR


lang = "hi"
normalizer_factory = IndicNormalizerFactory()
normalizer = normalizer_factory.get_normalizer(lang)
def process_sent(sent, lang = "hi"):    
    normalized = normalizer.normalize(sent)    
    processed = ' '.join(trivial_tokenize(normalized, lang))    
    return processed



def process_vardial():
    inpath = "{}/vardial2018/dataset/".format(MONO_DATADIR)
    outpath = "{}/vardial2018/formatted/".format(MONO_DATADIR)

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    files = ["dev.txt", "train.txt", "gold.txt"]
    langs = {"BHO":"bhojpuri", "MAG":"magahi", "AWA":"awadhi", "HIN":"hindi", "BRA":"brajbhasha"}

    for lang_code in langs:
        lang_outpath = outpath + lang_code.lower() + ".txt"
        out_lang = open(lang_outpath, "w", encoding="utf-8")
        for file in files:
            with open(inpath+file, "r", encoding="utf-8") as f:
                data = f.read().split("\n")
            for line in data:
                if lang_code in line:
                    line = " ".join([token for token in line.strip().split() if token != lang_code])
                    # line = process_sent(line)
                    out_lang.write(line+"\n")
        
        out_lang.close()


def process_mai_wikipedia_2021_10K():
    inpath = "{}/mai_wikipedia_2021_10K/mai_wikipedia_2021_10K-sentences.txt".format(MONO_DATADIR)
    outpath = "{}/mai_wikipedia_2021_10K/formatted/".format(MONO_DATADIR)

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    lang_outpath = outpath + "mai" + ".txt"
    with open(inpath, "r", encoding="utf-8") as in_f, \
        open(lang_outpath, "w", encoding="utf-8") as out_f:
        data = in_f.read().split("\n")
        for line in data:
            line = " ".join(line.split()[1:])
            out_f.write(line)
            out_f.write("\n")

def process_indiccorp():
    inpath = "{}/indiccorp/data/hi/hi.txt".format(MONO_DATADIR)
    # outpath = "{}/indiccorp/formatted/".format(MONO_DATADIR)
    outpath = "../data/monolingual/all/"

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    lang_outpath = outpath + "hin2" + ".txt"

    # with gzip.open(inpath, "rb") as in_f, gzip.open(lang_outpath, 'wb') as out_f:
    with open(inpath, "r", encoding="utf-8") as in_f, open(lang_outpath, 'w', encoding="utf-8") as out_f:
        i=0
        for line in tqdm(in_f):
            # i+=1
            # if i<0:
            #     continue
            # if i>=1000000:
            #     break
            # line = line.split("\t")[2]
            # print(line)
            line = process_sent(line)
            out_f.write(line.strip("\n") + "\n")

# process_vardial()
# process_mai_wikipedia_2021_10K()
process_indiccorp()






