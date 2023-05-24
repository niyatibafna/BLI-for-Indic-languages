#!/usr/bin/env python3

import os
import unicodedata

from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

lang = "hi"
normalizer_factory = IndicNormalizerFactory()
normalizer = normalizer_factory.get_normalizer(lang)
def process_sent(sent, lang = "hi"):    
    normalized = normalizer.normalize(sent)    
    processed = ' '.join(trivial_tokenize(normalized, lang))    
    return processed


def filter_cjk(text):
    """Filter out Chinese, Japanese, and Korean characters from a string."""
    filtered_text = ''
    bad_unicode_names = ["hiragana", "hangul", "tangut", "katakana", "yi", "cjk"]
    for char in text:
        bad_char = False # If try blocks fails, we don't want to filter the char
        try:
            for bad_name in bad_unicode_names:
                if bad_name in unicodedata.name(char).casefold():
                    bad_char = True
        except ValueError:
            pass
        if not bad_char:
            filtered_text += char

    return filtered_text

MONO_DATADIR = "../data/raw_monolingual/"
COLLATED = "../data/monolingual/all/"

if not os.path.isdir(COLLATED):
    os.mkdir(COLLATED)

def collate_data(lang, source_set: list = None):

    if lang == "bho":

        sources = [
        "{}/bhojpuri_monolingual_corpus/monolingual.bho".format(MONO_DATADIR), 
        "{}/loresmt/mono.loresmt-2020.bho".format(MONO_DATADIR),
        "{}/vardial2018/formatted/bho.txt".format(MONO_DATADIR)]

    if lang == "awa":
        sources = [ 
        "{}/vardial2018/formatted/awa.txt".format(MONO_DATADIR)]

    if lang == "mag":
        sources = [
        "{}/loresmt/mono.loresmt-2020.mag".format(MONO_DATADIR),
        "{}/vardial2018/formatted/mag.txt".format(MONO_DATADIR)]

    if lang == "hin":
        sources = [
            "{}/indiccorp/data/hi/hi.txt".format(MONO_DATADIR)
        # "{}/indiccorp/formatted/hin.txt".format(MONO_DATADIR)
        ]

    if lang == "bra":
        sources = [
        "{}/vardial2018/formatted/bra.txt".format(MONO_DATADIR)]

    if lang == "mai":
        sources = [
        "{}/mai_wikipedia_2021_10K/formatted/mai.txt".format(MONO_DATADIR),
        "{}/bmm_mai/formatted/mai.txt".format(MONO_DATADIR)
        ]

    if source_set is not None:
        sources = [source for source in sources if source in source_set]

    with open(COLLATED+lang+".txt", "w", encoding="utf-8") as out_f:
        for source in sources:
            with open(source, "r", encoding="utf-8") as in_f:
                # data = in_f.read().strip().split("\n")
                # print("SOURCE: {}, with # lines: {}".format(source, len(data)))
                for line in in_f:
                    line = line.rstrip("\n")
                    line = process_sent(line)
                    line = filter_cjk(line)
                    out_f.write(line)
                    out_f.write("\n")
        

if __name__=="__main__":
    langs = ["mai", "awa", "bra", "mag", "bho", "hin",]
    # langs = ["mai"]
    for lang in langs:
        print("Currently processing: {}".format(lang))
        collate_data(lang = lang)

