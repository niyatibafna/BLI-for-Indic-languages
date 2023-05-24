
LANGS = ["awa", "bho", "bra", "hin", "mag", "mai"]

import editdistance
# Define lambda function for normalized edit distance

ned = lambda s1, s2: editdistance.eval(s1, s2)/max(len(s1), len(s2))