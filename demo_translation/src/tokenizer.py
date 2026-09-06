"""
    分词器
"""
import abc
from abc import ABC

from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

from demo_translation.src.config import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

class BaseTokenizer(ABC):
    def __init__(self, vocab_list):
        self.__vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.id2word = vocab_list
        self.word2id = { word:index for index, word in enumerate(self.__vocab_list)}

        self.unk_id = self.word2id[UNK_TOKEN]
        self.pad_id = self.word2id[PAD_TOKEN]
        self.sos_id = self.word2id[SOS_TOKEN]
        self.eos_id = self.word2id[EOS_TOKEN]

    @classmethod
    @abc.abstractmethod
    def tokenize(cls, sentence):
        pass

    def encode(self, sentence):
        """
        把句子转换为词ID
        :param sentence:  需要转码的句子
        :return: ID 列表
        """
        ids = []
        for token in self.tokenize(sentence):
            ids.append(self.word2id.get(token, self.unk_id))
        return ids

    def decode(self, ids):
        tokens = []
        for index in ids:
            tokens.append(self.id2word[index])
        return tokens

    @classmethod
    def create_from_vocab_file(cls, vocab_file):
        vocab_list = []
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                vocab_list.append(line.strip())
        return cls(vocab_list)

    @classmethod
    def build_vocab_file(cls, sentence_list, vocab_file):
        vocab_set = set()
        for sentence in sentence_list:
            tokens = cls.tokenize(sentence)
            vocab_set.update(tokens)

        vocab_list = [UNK_TOKEN,
                      PAD_TOKEN,
                      SOS_TOKEN,
                      EOS_TOKEN] + list(vocab_set)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

class EnTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, sentence):
        return TreebankWordTokenizer().tokenize(sentence)

class ZhTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, sentence):
        return list(sentence)