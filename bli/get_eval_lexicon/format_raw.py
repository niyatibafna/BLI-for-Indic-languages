
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

lang = "hi"
def process_sent(sent, lang = "hi"):     
    processed = " ".join(trivial_tokenize(sent, lang))    
    return processed

# %%
HIN_FILE = "data/all/mag_parallel.hin"
TARGET_FILE = "data/all/all.mag"
OUTPUT = "data/formatted/mag_hin.txt"


hin = open(HIN_FILE, "r").read().split("\n")
target = open(TARGET_FILE, "r").read().split("\n")



# %%

output = list()
for idx in range(len(hin)):
    out_sent = process_sent(target[idx]) + " ||| " + process_sent(hin[idx])
    output.append(out_sent)

# %%
output[7]
# %%
with open(OUTPUT, "w") as f:
    f.write("\n".join(output))

# %%
