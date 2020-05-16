import regex as re
import gensim
import numpy as np
import torch
from sklearn.utils import shuffle

np.random.seed(15)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PreprocessData():
    def __init__(self, w2v_path, use_pretrained_vector=False):
        # Word2vec Location of local Machine
        # '../../../../data/google.news.word2vec/GoogleNews-vectors-negative300.bin'
        self.word2vec = None
        if use_pretrained_vector:
            print('Loading Word2vec')
            self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.re_word_tokenizer = re.compile(r"\w+", re.I)
        self.reset()

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def set_maximum_sentence_length(self, sentences):
        sentences_tok = self.tokenize_sentences(sentences)
        self.max_sen_len = np.max([len(s) for s in sentences_tok])

    def reset(self):
        self.dictionary = {}
        self.class2index = {}
        self.index2class = {}
        self.classCount = 0
        self.word2index = {'unk': 0}
        self.wordcount = {'unk': 0}
        self.index2word = {0: 'unk'}
        self.wordCount = 0
        self.wordCount_w2v = 0
        self.weights = None

    def tokenize_sentences(self, sentences):
        if self.dataset_name == 'SST1' or self.dataset_name == 'SST2':
            return [self.clean_str_sst(sen).split(' ') for sen in sentences]
        else:
            return [self.clean_str(sen).split(' ') for sen in sentences]

    def get_average_sentence_length(self, sentences):
        sentences_tok = self.tokenize_sentences(sentences)
        return np.mean([len(s) for s in sentences_tok])

    def update_dict(self, sent):
        for word in sent:
            if not word.lower() in self.word2index:
                self.wordCount += 1
                self.word2index[word.lower()] = self.wordCount
                self.index2word[self.wordCount] = word.lower()
                self.wordcount[word.lower()] = 0
            self.wordcount[word.lower()] += 1

    def train_dictionary(self, sentences, use_pretrained_vector=False):
        sentences_tok = self.tokenize_sentences(sentences)
        for sent_tok in sentences_tok:
            self.update_dict(sent_tok)

        if use_pretrained_vector:
            self.weights = np.zeros((self.wordCount + 1, self.word2vec.vector_size), np.float)
            for i in self.index2word:
                if self.index2word[i] in self.word2vec:
                    self.weights[i, :] = self.word2vec[self.index2word[i]]
                    self.wordCount_w2v += 1
                elif self.wordcount[self.index2word[i]] >= 5:
                    self.weights[i, :] = np.random.uniform(-0.25, 0.25, self.word2vec.vector_size)
                else:
                    pass

            # Old method the weights were directly picked up from word2vec,
            # weight values were not matching in th end thus skipped to self retaining weights
            # self.word2index = {token: token_index for token_index, token in enumerate(self.word2vec.index2word)}
            # self.index2word = {token_index: token for token_index, token in enumerate(self.word2vec.index2word)}
            # self.update_wordCount_w2v()

    def train_classes(self, classes):
        for cl in np.unique(classes):
            if cl not in self.class2index:
                self.class2index[cl] = self.classCount
                self.index2class[self.classCount] = cl
                self.classCount += 1

    def sent2Index(self, sentences):
        sentIndexed = []
        sentences_tok = self.tokenize_sentences(sentences)
        for sent_tok in sentences_tok:
            sentIndx = []
            for w in sent_tok:
                if w.lower() in self.word2index:
                    sentIndx.append(self.word2index[w.lower()])
                else:
                    sentIndx.append(self.word2index['unk'])

            if len(sentIndx) < self.max_sen_len:
                sentIndx = sentIndx + ([self.word2index['unk']] * (self.max_sen_len - len(sentIndx)))
            sentIndexed.append(sentIndx)

            ## As per paper we initially used variable sentence length
            ## adding 'unk' parameter only where it was necessary to achieve min filter
            ## length. While similar accuracy is achieved by the method it lacks speed.
            ## Missing out matrix multiplication as the modelling layer
            # if len(sentIndx) < 5:
            #     sentIndx = sentIndx + ([self.word2index['unk']] * (5 - len(sentIndx)))
            # sentIndexed.append(torch.LongTensor(sentIndx).to(device))

        return torch.LongTensor(sentIndexed).to(device)

    def clean_str(self, sent):
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " ( ", sent)
        sent = re.sub(r"\)", " ) ", sent)
        sent = re.sub(r"\?", " ? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
        return sent.strip().lower()

    def clean_str_sst(self, sent):
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
        return sent.strip().lower()

    def class2Index(self, classList):
        return torch.LongTensor([self.class2index[c] for c in classList]).to(device)

    def proprocess_data(self, x, y):
        x, y = shuffle(x, y, random_state=17)
        x = self.sent2Index(x)
        y = self.class2Index(y)

        return x, y