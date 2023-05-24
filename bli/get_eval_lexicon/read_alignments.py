#!/usr/bin/env python3
import string
from collections import defaultdict
import sys
import json

f = open(sys.argv[1], "r").read().split("\n")
alignments = open(sys.argv[2], "r").read().split("\n")

source, target = list(), list()
f = [line for line in f if line != ""]
for line in f:
    if "|||" not in line:
        continue

    # FOR HINDI AS SOURCE
    # source.append(line.split("|||")[1].strip())
    # target.append(line.split("|||")[0].strip())

    source.append(line.split("|||")[0].strip())
    target.append(line.split("|||")[1].strip())


dictionary = defaultdict(lambda: defaultdict(lambda: 0))

for idx in range(len(alignments)):
    # print(source[idx])
    # print(target[idx])
    if idx >= min(len(source), len(target), len(alignments)):
        continue
    source_words = source[idx].strip().split()
    target_words = target[idx].strip().split()
    align = [(int(x.split("-")[0]), int(x.split("-")[1])) for x in alignments[idx].strip().split()]
    for a in align:
        sw, tw = source_words[a[0]], target_words[a[1]]
        # FOR HINDI AS SOURCE
        # sw, tw = source_words[a[1]], target_words[a[0]]
        dictionary[sw][tw] += 1


# for word in dictionary:
#     print(word)
#     print(dict(dictionary[word]))
#     print("\n\n")
with open(sys.argv[3], "w") as f:
    json.dump(dictionary, f, indent = 2, ensure_ascii = False)
