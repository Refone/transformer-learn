"""
    分词器
"""
import abc
from abc import ABC

from nltk import TreebankWordTokenizer, TreebankWordDetokenizer

from config import *

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

    def encode(self, sentence, mark=False):
        """
        把句子转换为词ID
        :param sentence:  需要转码的句子
        :param mark:    是否在前后加上 <sos> <eos>
        :return: ID 列表

        之所以要在这个阶段加 tag,一方面是性能优化,
        另一方面,可以避免处理 <eos> 和 <pad> 谁先谁后的问题
        """
        # 1. 分词,得到 token 列表
        tokens = self.tokenize(sentence)

        # 2. 将 token 转化为 id
        ids = [ self.word2id.get(token, self.unk_id) for token in tokens ]

        if mark:
            ids = [self.sos_id] + ids + [self.eos_id]

        return ids

    @abc.abstractmethod
    def decode(self, ids):
        pass

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

        vocab_list = [PAD_TOKEN,
                      UNK_TOKEN,
                      SOS_TOKEN,
                      EOS_TOKEN] + sorted(vocab_set)

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

class EnTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, sentence):
        return cls.tokenizer.tokenize(sentence)

    def decode(self, ids):
        tokens = [ self.id2word[index] for index in ids ]
        sentence = self.detokenizer.detokenize(tokens)
        return sentence

class ZhTokenizer(BaseTokenizer):
    @classmethod
    def tokenize(cls, sentence):
        return list(sentence)

    def decode(self, ids):
        tokens = []
        for index in ids:
            tokens.append(self.id2word[index])
        return ''.join(tokens)

if __name__ == '__main__':
    # --- UNIT-TEST ---
    from tokenizer import EnTokenizer, ZhTokenizer

    en_tokenizer = EnTokenizer.create_from_vocab_file(EN_VOCAB_FILE)
    zh_tokenizer = ZhTokenizer.create_from_vocab_file(ZH_VOCAB_FILE)

    en_text = "I am a student."
    zh_text = "我是一名学生。"

    en_tokens = en_tokenizer.tokenize(en_text)
    zh_tokens = zh_tokenizer.tokenize(zh_text)

    print(f'en text -> tokens: {en_text} -> {en_tokens}')
    print(f'zh text -> tokens: {zh_text} -> {zh_tokens}')

    en_ids = en_tokenizer.encode(en_text, mark=True)
    zh_ids = zh_tokenizer.encode(zh_text, mark=True)

    print(f'en text -> ids: {en_text} -> {en_ids}')
    print(f'zh text -> ids: {zh_text} -> {zh_ids}')

    en_decode = en_tokenizer.decode(en_ids)
    zh_decode = zh_tokenizer.decode(zh_ids)

    print(f'en ids -> text: {en_ids} -> {en_decode}')
    print(f'zh ids -> text: {zh_ids} -> {zh_decode}')
