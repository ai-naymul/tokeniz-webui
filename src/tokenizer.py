from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast


## https://www.perplexity.ai/search/i-want-to-build-a-webui-based-_MpcRIypRpqp2h1fld_NEw
class TrainTokeniers():
    def __init__(self):
        pass
    # train bpe trainer 
    def train_bpe(self, files, special_tokens, vocab_size, min_freq):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens = special_tokens, vocab_size = vocab_size, min_frequency = min_freq)
        tokenizer.train(files, trainer)
        return tokenizer
    
    # https://huggingface.co/docs/tokenizers/en/api/models#tokenizers.models.Unigram
    # https://huggingface.co/docs/tokenizers/en/api/trainers#tokenizers.trainers.WordLevelTrainer
    def train_unigram(self, files, special_tokens, vocab_size):
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens, vocab_size = vocab_size)
        tokenizer.train(files, trainer)
        return tokenizer

    def train_wordlevel(self, files, special_tokens, vocab_size, min_freq):

        tokenizer = Tokenizer(WordLevel(unk_token=["UNK"]))
        trainer = WordLevelTrainer(special_tokens, vocab_size=vocab_size, min_frquency=min_freq)
        tokenizer.train(files=files, trainer=trainer)
        return tokenizer
    
    def train_wordpiece(self, files, special_tokens, vocab_size, min_freq):
        tokenizer = Tokenizer(WordPiece(unk_token=["UNK"]))
        trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size= vocab_size, min_frequency= min_freq)
        tokenizer.train(files=files, trainer=trainer)
        return tokenizer

    # loading the tokenizer for uses
    def load_tokenizer(json_file):
        fast_tokenizer = PreTrainedTokenizerFast(json_file)
        return fast_tokenizer



