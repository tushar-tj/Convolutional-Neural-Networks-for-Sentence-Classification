from load_data import LoadData
from preprocess_data import PreprocessData
from model_data import CNNClassificationModel
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pandas as pd
import time
import datetime
import sys
import numpy as np
import torch
import argparse
import os
import pickle

np.random.seed(15)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def evaluate(x, y, model, criterion):
    model.eval() # setting model to eval mode
    y_pred = model(x)
    loss = criterion(y_pred, y)
    accuracy = accuracy_score(y, y_pred.argmax(-1))
    return loss.item(), accuracy

def get_batch(x, y, batch_size=50):
    x, y = shuffle(x, y, random_state=13)
    start_index, end_index = 0, 0
    data_batches = []
    while end_index < len(x):
        end_index = (start_index + batch_size) if (start_index + batch_size) < len(x) else len(x)
        data_batches.append((x[start_index:end_index], y[start_index:end_index]))
        start_index = start_index + batch_size
    return data_batches

def train(x_train, y_train, x_test, y_test, model, optimizer, criterion, model_name='model',
          model_store=False, batch_size=50, epochs=100, log_iter=10):

    data_batches = get_batch(x_train, y_train, batch_size)
    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []
    start_time = time.time()
    for epoch in range(epochs):
        # Setting model intot training mode
        model.train()  # setting model in train mode
        for batch_num, batch in enumerate(data_batches):
            x, y = batch[0], batch[1]
            y_pred = model(x)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()

            # This piece is to constrain the normalized weight ||w|| of
            # the output linear layer be constrained at 3
            # l2_weights = torch.norm(model.linear.weight).item()
            # if l2_weights > 3:
            #     model.linear.parameters = model.linear.weight/l2_weights
            optimizer.step()

            sys.stdout.write('\rEpoch:{} | Batch:{} | Time Running: {}'.format(epoch,
                                                                               batch_num,
                                                                               datetime.timedelta(seconds=np.round(
                                                                                   time.time() - start_time, 0))))

        trainloss, trainacc = evaluate(x_train, y_train, model, criterion)
        testloss, testacc = evaluate(x_test, y_test, model, criterion)

        if epoch > log_iter and testacc > np.max(test_accuracy) and model_store is True:
            torch.save(model, './models/' + model_name + '.torch')

        train_loss.append(trainloss)
        train_accuracy.append(trainacc)
        test_loss.append(testloss)
        test_accuracy.append(testacc)

        if epoch % log_iter == 0:
            print(' Train Acc {:.4f}, Train Loss {:.4f}, Test Acc {:.4f}, Test Loss {:.4f}'.format(trainacc,
                                                                                                   trainloss,
                                                                                                   testacc,
                                                                                                   testloss))

    print(' Best Test Accuracy {:.4f}'.format(np.max(test_accuracy)))
    return train_loss, train_accuracy, test_loss, test_accuracy

parser = argparse.ArgumentParser(description='CNN Based Text classifier CNN - Training')
# Data Locations
parser.add_argument('-dataset', type=str, default='MR',
                    help='Name of the Dataset to use')
parser.add_argument('-dataset_path', type=str, default=None,
                    help='Location to the Dataset')
parser.add_argument('-word2vec_path', type=str, default=None,
                    help='Location to GoogleNews-vectors-negative300.bin file')

# Model iterations and size control
parser.add_argument('-epochs', type=int, default=25,
                    help='Number of Epochs to train')
parser.add_argument('-epochs_log', type=int, default=5,
                    help='Log Accuracy after every X Epochs')
parser.add_argument('-batch_size', type=int, default=50,
                    help='Batch Size to use while training')

# Hyperparameters for Tuning
parser.add_argument('-optimizer', type=str, default='Adam',
                    help='Select optimizer from "Adam" or "Adadelta"')
parser.add_argument('-embedding_size', type=int, default=300,
                    help='Embedding size to be used in case of self trained embedding only, '
                         'redundant is using pretrained vector')
parser.add_argument('-lr', type=float, default=0.001,
                    help='Learning rate to use while training')

# Running Different variations of model
parser.add_argument('-use_pretrained_vector', type=str2bool, nargs='?',const=True, default=False,
                    help='To use Pretrained vector (word2vec) or not')
parser.add_argument('-keep_embedding_static', type=str2bool, nargs='?',const=True, default=False,
                    help='Would like to train/adjust the Embedding or not')
parser.add_argument('-use_multi_channel', type=str2bool, nargs='?', const=True, default=False,
                    help='Use multichannel or not')

# Storing Logs and Model
parser.add_argument('-model_name', type=str, default='model',
                    help='Provide a name to the model, use the names in the paper for logging')
parser.add_argument('-save_model', type=str2bool, nargs='?', const=True, default=False,
                    help='Would like to store the model or not, model is '
                         'stored by dataset name and model name arguments provided')
parser.add_argument('-log_results', type=str2bool, nargs='?', const=True, default=False,
                    help='Would like to log the final results of the model')

args = parser.parse_args()

print ('Arguments Loaded')
print (args)
print ('-' * 20)

data_loader = LoadData(dataset_name=args.dataset, dataset_path=args.dataset_path)
data_preprocessor = PreprocessData(w2v_path=args.word2vec_path, use_pretrained_vector=args.use_pretrained_vector)
data_preprocessor.reset()
data_preprocessor.set_dataset_name(args.dataset)
data_preprocessor.set_maximum_sentence_length(data_loader.data[0]['x_train'] + data_loader.data[0]['x_test'])
data_preprocessor.train_dictionary(data_loader.data[0]['x_train'] + data_loader.data[0]['x_test'],
                                   use_pretrained_vector=args.use_pretrained_vector)
data_preprocessor.train_classes(data_loader.data[0]['y_train'])


print ('\n\nDataset Name - {}'.format(args.dataset))
print ('Number of Classes - {}'.format(data_preprocessor.classCount))
print ('Average Length of Sentences - {}'.format(np.round(data_preprocessor.get_average_sentence_length(data_loader.data[0]['x_train'] +
                                                                                                 data_loader.data[0]['x_test']), 0)))
print ('Max Length of Sentence - {}'.format(data_preprocessor.max_sen_len))
print ('Dataset Size - {}'.format(len(data_loader.data[0]['x_train'] + data_loader.data[0]['x_test'])))
print ('Number of Words - {}'.format(data_preprocessor.wordCount))
if args.use_pretrained_vector:
    print ('Number of Words in Word2Vec - {}'.format(data_preprocessor.wordCount_w2v))

print ('Test Data Size - {}'.format('CV' if len(data_loader.data) > 1 else len(data_loader.data[0]['x_test'])))

if args.save_model:
    with open('./models/' + args.model_name + '.preprocessor', 'wb') as f:
        pickle.dump(data_preprocessor, f)

if not os.path.isdir('models'):
    os.mkdir('models')

if not os.path.isdir('results'):
    os.mkdir('models')

accuracy = []
for d in data_loader.data:
    x_train = data_preprocessor.sent2Index(d['x_train'])
    y_train = data_preprocessor.class2Index(d['y_train'])

    x_test = data_preprocessor.sent2Index(d['x_test'])
    y_test = data_preprocessor.class2Index(d['y_test'])

    model = CNNClassificationModel(use_pretrained_vector=args.use_pretrained_vector,
                                   pretrained_vector_weight=data_preprocessor.weights,
                                   word_count=data_preprocessor.wordCount + 1,
                                   embedding_size=args.embedding_size,
                                   number_of_classes=data_preprocessor.classCount,
                                   keep_embeddings_static=args.keep_embedding_static,
                                   use_multi_channel=args.use_multi_channel).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    if args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, eps=1e-06, weight_decay=1e-03)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loss, train_accuracy, test_loss, test_accuracy = train(x_train, y_train, x_test, y_test,
                                                                 model, optimizer, criterion,
                                                                 epochs=args.epochs,
                                                                 log_iter=args.epochs_log,
                                                                 model_name=args.dataset + '_' + args.model_name,
                                                                 model_store=args.save_model,
                                                                 batch_size=args.batch_size)

    accuracy.append(test_accuracy[-1])

print('\nTest Accuracy {}'.format(np.mean(accuracy)))

# logging model results
if args.log_results:
    if not os.path.isfile('./results/' + args.dataset + '.csv'):
        df = pd.DataFrame({'Date': [], 'Model Type': [], 'Test Accuracy': [], 'Parameters': []})
    else:
        df = pd.read_csv('./results/' + args.dataset + '.csv')

    df = pd.concat([df, pd.DataFrame({'Date': [datetime.datetime.now().strftime('%d %b %Y')],
                                      'Model Type': [args.model_name],
                                      'Test Accuracy': [np.mean(accuracy)],
                                      'Parameters': [args]})])

    df.to_csv('./results/' + args.dataset + '.csv', index=False)