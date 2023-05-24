#%%
from pyiwn import iwn
import random
import json
from collections import defaultdict

#%%

SOURCE_LANG = "mar"
TARGET_LANG = "hin"

lang_to_iwn = {
    "mar": iwn.Language.MARATHI,
    "hin": iwn.Language.HINDI,
    "nep": iwn.Language.NEPALI,
}

s_iwn = iwn.IndoWordNet(lang_to_iwn[SOURCE_LANG])
t_iwn = iwn.IndoWordNet(lang_to_iwn[TARGET_LANG])

s_word2ids = s_iwn._synset_idx_map
t_word2ids = t_iwn._synset_idx_map

#%%
# Invert word2ids lexicon for source and target

s_id2words = defaultdict(lambda: list())
for word, ids in s_word2ids.items():
    for id in ids:
        # if len(word.split()) == 1:
        s_id2words[id].append(word.split()[0])

t_id2words = defaultdict(lambda: list())
for word, ids in t_word2ids.items():
    for id in ids:
        # if len(word.split()) == 1:
        t_id2words[id].append(word.split()[0])

#%%
## GET LEXICON FOR ALL POSSIBLE WORDS

s2hin_lexicon = defaultdict(lambda: dict())

for id, s_words in s_id2words.items():
    if id in t_id2words:
        t_words = t_id2words[id]
        for s_word in s_words:
            for t_word in t_words:
                s2hin_lexicon[s_word][t_word] = 1

#%%
output_path = "../get_eval_lexicon/lexicons/target2hin/{}2{}.json".format(SOURCE_LANG, TARGET_LANG)
with open(output_path, "w") as f:
    json.dump(s2hin_lexicon, f, indent=2, ensure_ascii=False)

print("LENGTH OF LEXICON", len(s2hin_lexicon))

# print(len(s2hin_lexicon))
# list(s2hin_lexicon.items())[100:120]

# # %%
# # Take 5000 random words from the lexicon
# random.seed(42)
# sample = random.sample(list(s2hin_lexicon.items()), 5000)
# sample_dict = dict(sample)

# # %%
# sample_dict

# # %%
