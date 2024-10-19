from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpreTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer
from transformers import PreTrainedTokenizerFast

class TrainTokeniers():
    def __init__(self):
        pass
    # train bpe trainer 
    def train_bpe(self, files, special_tokens, vocab_size, min_freq):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(special_tokens = special_tokens, vocab_size = vocab_size, min_freq = min_frquency)
        tokenizer.train(files, trainer)
        return tokenizer
    # https://huggingface.co/docs/tokenizers/en/api/models#tokenizers.models.Unigram
    # https://huggingface.co/docs/tokenizers/en/api/trainers#tokenizers.trainers.WordLevelTrainer
    def train_unigram(self, files, special_tokens, vocab_size, min_freq):
        pass



