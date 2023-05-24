from cProfile import label
from conllu import parse
from sklearn.model_selection import train_test_split

SEED = 42

def get_data(DATAPATH, format = "conllu", feature = "xpos", SEED = 42, split = True):

    '''Reads data from different formats'''

    with open(DATAPATH, "r") as f:
        data = f.read()

    if DATAPATH.endswith("conllu"):

        sentences = parse(data)

        data = [" ".join([meta["form"] for meta in sentence]) for idx, sentence in enumerate(sentences)]
        labels = [" ".join([meta[feature] for meta in sentence]) for idx, sentence in enumerate(sentences)]

    if DATAPATH.endswith("conll"):

        sentences = data.split("\n\n")
        
        data = list()
        labels = list()
        for sent in sentences:
            data.append(" ".join([word.split()[0] for word in sent.split("\n") if len(word.split()) >= 2]))
            labels.append(" ".join([word.split()[1] for word in sent.split("\n") if len(word.split()) >= 2]))

        print("Total # sentences", len(data))

    if DATAPATH.endswith("bmm"):
        tagset = \
            {'SYM', 'QC', 'NNP', 'PSP',\
                 'ECH', 'RDP', 'RP', \
                'DEM', 'CC', 'VAUX', 'INTF', 'UNK', \
                'JJ', 'NEG', 'XC', 'INJ', \
                'VM', 'NN', 'WQ', \
                'QO', 'PRP', 'NST', 'CL', \
                'QF', 'RB'}
        with open(DATAPATH, "r") as f:
            sentences = f.read().split("\n")

        data = list()
        labels = list()
        start_sent = False

        for line in sentences:
            if "((" in line or "))" in line:
                continue
            if line.startswith("</Sentence"):
                assert len(curr_sent) == len(curr_tags)
                data.append(" ".join(curr_sent))
                labels.append(" ".join(curr_tags))
                start_sent = False

            if start_sent:
                try:
                    # print(line.split("\t"))
                    if line.split("\t")[2] in tagset:
                        curr_sent.append(line.split("\t")[1])
                        curr_tags.append(line.split("\t")[2])
                except:
                    print(line)
            if line.startswith("<Sentence"):
                start_sent = True
                curr_sent = list()
                curr_tags = list()
        
        print(len(data))



    if split == False:
        return data, labels

    train_data, dev_data, train_labels, dev_labels = train_test_split(data, labels, test_size = 0.2,\
    random_state = SEED)
    dev_data, test_data, dev_labels, test_labels = train_test_split(dev_data, dev_labels, test_size = 0.5,\
    random_state = SEED)

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels

def get_data_lid(DATADIR, LANGS:list,  SEED = 42, split = True, max_lines = 50000):
    '''Reads monolingual data and returns train, dev and test splits, labels are the language names'''
    print("Reading data from", DATADIR, "for languages", LANGS, "with max lines", max_lines, "per language")

    data = list()
    labels = list()
    for lang in LANGS:
        print(lang)
        DATAPATH = DATADIR + "/" + lang + ".txt"
        sents = 0
        with open(DATAPATH, "r") as f:
            for line in f:
                if len(line.split()) <= 2:
                    continue
                data.append(line)
                labels.append(lang)
                sents += 1
                if sents >= max_lines:
                    print("Max lines reached for", lang)
                    print(len(data), len(labels))
                    break
        print("Here for lang", lang, "sents", sents)
        print(len(data), len(labels))
                
    print("Total # sentences", len(data))

    if split == False:
        return data, labels
    
    train_data, dev_data, train_labels, dev_labels = train_test_split(data, labels, test_size = 0.2,\
    random_state = SEED)
    dev_data, test_data, dev_labels, test_labels = train_test_split(dev_data, dev_labels, test_size = 0.5,\
    random_state = SEED)

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels

def get_data_mt(DATADIR, lang, SEED = 42, split = True, max_lines = 50000):

    hin_files = ["{}/{}2hi.test.hi".format(DATADIR, lang), "{}/hi2{}.test.hi".format(DATADIR, lang)]
    lrl_files = ["{0}/{1}2hi.test.{1}".format(DATADIR, lang), "{0}/hi2{1}.test.{1}".format(DATADIR, lang)]

    hrl_sents = list()
    lrl_sents = list()
    for hin_file, lrl_file in zip(hin_files, lrl_files):
        with open(hin_file, "r") as f:
            for line in f:
                hrl_sents.append(line)
        with open(lrl_file, "r") as f:
            for line in f:
                lrl_sents.append(line)
    
    assert len(hrl_sents) == len(lrl_sents)
    print("Total # parallel sentences", len(hrl_sents))

    return hrl_sents, lrl_sents



if __name__=="__main__":

    DATAPATH = "../data/raw_parallel/loresmt/"
    # DATAPATH  = "../data/eval_POS/mag.nsurl.bis.conllu"
    # train_data, train_labels, dev_data, dev_labels, test_data, test_labels = \
    # get_data(DATAPATH, format="conllu")
    # print(train_data[:2])
    # print("TAGSET", len(set([tag for sent in train_labels for tag in sent.split()])))
    # print("TAGSET", set([tag for sent in train_labels for tag in sent.split()]))



