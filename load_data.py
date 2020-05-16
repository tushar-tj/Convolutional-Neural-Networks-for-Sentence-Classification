from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class LoadData():
    def __init__(self, dataset_name, dataset_path=None):

     # mr='../../../../data/MR-rt-polaritydata/rt-polaritydata/',
     # sst='../../../../data/SST-stanfordSentimentTreebank/',
     # subj='../../../../data/Subj-rotten_imdb/',
     # trec='../../../../data/TREC-Trec6/',
     # cr='../../../../data/CR-customer review data/',
     # mpqa='../../../../data/MRQA-database.mpqa.1.2/'):

        self.dataset_path = dataset_path
        self.data = []
        assert dataset_name in ['MR', 'SST1', 'SST2', 'Subj', 'TREC', 'CR', 'MPQA']
        if dataset_name == 'MR':
            self.load_mr_data()

        if dataset_name == 'SST1':
            self.load_sst1_data()

        if dataset_name == 'SST2':
            self.load_sst2_data()

        if dataset_name == 'Subj':
            self.load_subj_data()

        if dataset_name == 'TREC':
            self.load_trec_data()

        if dataset_name == 'CR':
            self.load_cr_data()

        if dataset_name == 'MPQA':
            self.load_mpqa_data()

    def load_data_in_kfolds(self, x, y, folds=10):
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(x)
        for train_index, test_index in kf.split(x):
            self.data.append({'x_train': [x[i] for i in train_index],
                              'y_train': [y[i] for i in train_index],
                              'x_test': [x[i] for i in test_index],
                              'y_test': [y[i] for i in test_index]})

    def load_mr_data(self):
        with open(self.dataset_path + '/rt-polarity.pos', 'r', encoding='utf-8', errors='ignore') as f:
            pos_statements = f.read().split('\n')
        with open(self.dataset_path + '/rt-polarity.neg', 'r', encoding='utf-8', errors='ignore') as f:
            neg_statements = f.read().split('\n')
        x = pos_statements + neg_statements
        y = ['pos'] * len(pos_statements) + ['neg'] * len(neg_statements)
        self.load_data_in_kfolds(x, y)

    def load_sst1_data(self):
        sentences = pd.read_csv(self.dataset_path + '/datasetSentences.txt', sep='\t')
        sentences_type = pd.read_csv(self.dataset_path + '/datasetSplit.txt', sep=',')
        phase_labels = pd.read_csv(self.dataset_path + '/sentiment_labels.txt', sep='|')
        phase_dict = pd.read_csv(self.dataset_path + '/dictionary.txt', sep='|',
                                 header=None,
                                 names=['phase', 'phase_id'])

        data = pd.merge(sentences, sentences_type, on='sentence_index', how='left')
        data = pd.merge(data, phase_dict, left_on='sentence', right_on='phase', how='left')
        data = pd.merge(data, phase_labels, left_on='phase_id', right_on='phrase ids', how='left')

        sentiment = []
        for index, dp in data.iterrows():
            if dp['sentiment values'] <= 0.20:
                sentiment.append('very negative')
            elif dp['sentiment values'] > 0.20 and dp['sentiment values'] <= 0.40:
                sentiment.append('negative')
            elif dp['sentiment values'] > 0.40 and dp['sentiment values'] <= 0.60:
                sentiment.append('neutral')
            elif dp['sentiment values'] > 0.60 and dp['sentiment values'] <= 0.80:
                sentiment.append('positive')
            elif dp['sentiment values'] > 0.80:
                sentiment.append('very positive')
            else:
                sentiment.append(np.NaN)

        data['sentiment'] = sentiment
        data.dropna(subset=['sentence', 'sentiment'], inplace=True)
        self.data.append({'x_train': data[data.splitset_label.isin([1, 3])]['sentence'].tolist(),
                          'y_train': data[data.splitset_label.isin([1, 3])]['sentiment'].tolist(),
                          'x_test': data[data.splitset_label == 2]['sentence'].tolist(),
                          'y_test': data[data.splitset_label == 2]['sentiment'].tolist()})

    def load_sst2_data(self):
        sentences = pd.read_csv(self.dataset_path + '/datasetSentences.txt', sep='\t')
        sentences_type = pd.read_csv(self.dataset_path + '/datasetSplit.txt', sep=',')
        phase_labels = pd.read_csv(self.dataset_path + '/sentiment_labels.txt', sep='|')
        phase_dict = pd.read_csv(self.dataset_path + '/dictionary.txt', sep='|',
                                 header=None,
                                 names=['phase', 'phase_id'])

        data = pd.merge(sentences, sentences_type, on='sentence_index', how='left')
        data = pd.merge(data, phase_dict, left_on='sentence', right_on='phase', how='left')
        data = pd.merge(data, phase_labels, left_on='phase_id', right_on='phrase ids', how='left')

        sentiment = []
        for index, dp in data.iterrows():
            if dp['sentiment values'] <= 0.40:
                sentiment.append('negative')
            elif dp['sentiment values'] > 0.60:
                sentiment.append('positive')
            else:
                sentiment.append(np.NaN)

        data['sentiment'] = sentiment
        data.dropna(subset=['sentence', 'sentiment'], inplace=True)
        self.data.append({'x_train': data[data.splitset_label.isin([1, 3])]['sentence'].tolist(),
                          'y_train': data[data.splitset_label.isin([1, 3])]['sentiment'].tolist(),
                          'x_test': data[data.splitset_label == 2]['sentence'].tolist(),
                          'y_test': data[data.splitset_label == 2]['sentiment'].tolist()})

    def load_subj_data(self):
        with open(self.dataset_path + '/plot.tok.gt9.5000', 'r', encoding='utf-8', errors='ignore') as f:
            subjective_sentences = f.readlines()
        with open(self.dataset_path + '/quote.tok.gt9.5000', 'r', encoding='utf-8', errors='ignore') as f:
            objective_sentences = f.readlines()

        x = subjective_sentences + objective_sentences
        y = ['subj'] * len(subjective_sentences) + ['obj'] * len(objective_sentences)
        self.load_data_in_kfolds(x, y)

    def load_trec_data(self):
        data_train = pd.read_csv(self.dataset_path + '/train_5500.label.txt',
                                 sep=":", encoding='latin8', header=None, names=['Topic', 'Sentence'])
        data_test = pd.read_csv(self.dataset_path + '/TREC_10.label.txt',
                                sep=":", encoding='latin8', header=None, names=['Topic', 'Sentence'])

        self.data.append({'x_train': data_train['Sentence'].tolist(),
                          'y_train': data_train['Topic'].tolist(),
                          'x_test': data_test['Sentence'].tolist(),
                          'y_test': data_test['Topic'].tolist()})

    def load_cr_data(self):
        reviews = []
        product_type = []

        with open(self.dataset_path + '/Canon G3.txt',
                  'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()[11:]
            for text in data:
                reviews.append(text.strip().split('##')[-1].replace('[t]', ''))
                product_type.append('camera')

        with open(self.dataset_path + '/Apex AD2600 Progressive-scan DVD player.txt',
                  'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()[11:]
            for text in data:
                reviews.append(text.strip().split('##')[-1].replace('[t]', ''))
                product_type.append('mp3s etc')

        with open(self.dataset_path + '/Creative Labs Nomad Jukebox Zen Xtra 40GB.txt',
                  'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()[11:]
            for text in data:
                reviews.append(text.strip().split('##')[-1].replace('[t]', ''))
                product_type.append('mp3s etc')

        with open(self.dataset_path + '/Nikon coolpix 4300.txt',
                  'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()[11:]
            for text in data:
                reviews.append(text.strip().split('##')[-1].replace('[t]', ''))
                product_type.append('camera')

        with open(self.dataset_path + '/Nokia 6610.txt',
                  'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()[11:]
            for text in data:
                reviews.append(text.strip().split('##')[-1].replace('[t]', ''))
                product_type.append('mp3s etc')
        x = reviews
        y = product_type
        self.load_data_in_kfolds(x, y)

    def load_mpqa_data(self):
        text, sentiment = [], []
        with open(self.dataset_path + '/mpqa.neg.txt', 'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()
            text = text + data
            sentiment = sentiment + ['neg'] * len(data)

        with open(self.dataset_path + '/mpqa.pos.txt', 'r', encoding='utf-8', errors='ignore') as f:
            data = f.readlines()
            text = text + data
            sentiment = sentiment + ['pos'] * len(data)
        x = text
        y = sentiment
        self.load_data_in_kfolds(x, y)