from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from transformers import PreTrainedTokenizerFast, AutoTokenizer
import os, sys
sys.path.append("../")
from utils.variables import LANGS


def train_or_load_tokenizer(TOKENIZER_OUTPATH, \
    FILES = None, 
    vocab_size = 30_522):

    if not os.path.exists(TOKENIZER_OUTPATH) and FILES is not None:
        print("Training tokenizer...")
        special_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
        ]

        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(special_tokens=\
            special_tokens, \
            vocab_size = vocab_size)

        tokenizer.pre_tokenizer = Whitespace()
        print(FILES)

        tokenizer.train(FILES, trainer)

        tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[ \
            ("[CLS]", tokenizer.token_to_id("[CLS]")), \
            ("[SEP]", tokenizer.token_to_id("[SEP]"))],)

        tokenizer.save(TOKENIZER_OUTPATH)

        print("Tokenizer learnt and saved!")
        
    return load_tokenizer(TOKENIZER_OUTPATH)


def load_tokenizer(TOKENIZER_PATH, fast = True):

    print("Loading tokenizer...")
    if fast:
        if TOKENIZER_PATH == "Shushant/nepaliBERT":
            fast_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, model_max_length=512)
        else:
            fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_PATH, model_max_length=512)

        fast_tokenizer.bos_token="<s>"
        fast_tokenizer.eos_token="</s>"
        fast_tokenizer.sep_token="[SEP]"
        fast_tokenizer.cls_token="[CLS]"
        fast_tokenizer.unk_token="[UNK]"
        fast_tokenizer.pad_token="[PAD]"
        fast_tokenizer.mask_token="[MASK]"
                
        # fast_tokenizer.bos_token_id=0
        # fast_tokenizer.eos_token_id=2
        # fast_tokenizer.sep_token_id=2
        # fast_tokenizer.cls_token_id=0
        # fast_tokenizer.unk_token_id=3
        # fast_tokenizer.pad_token_id=1
        # fast_tokenizer.mask_token_id=4
                
        fast_tokenizer._bos_token="<s>"
        fast_tokenizer._eos_token="</s>"
        fast_tokenizer._sep_token="[SEP]"
        fast_tokenizer._cls_token="[CLS]"
        fast_tokenizer._unk_token="[UNK]"
        fast_tokenizer._pad_token="[PAD]"
        fast_tokenizer._mask_token="[MASK]"
                
        # fast_tokenizer._bos_token_id=0
        # fast_tokenizer._eos_token_id=2
        # fast_tokenizer._sep_token_id=2
        # fast_tokenizer._cls_token_id=0
        # fast_tokenizer._unk_token_id=3
        # fast_tokenizer._pad_token_id=1
        # fast_tokenizer._mask_token_id=4

        print("Done")

        return fast_tokenizer

    
def add_dialectid_tokens(tokenizer, dialect_ids: list = ["<"+lang+">" for lang in LANGS]):
    tokenizer.add_special_tokens({"additional_special_tokens": dialect_ids})
    return tokenizer
