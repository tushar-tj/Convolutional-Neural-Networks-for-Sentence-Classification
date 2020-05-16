# Project Title

This is an implementation of CNN for Sentence Classification by [Yoon Kim](https://www.aclweb.org/anthology/D14-1181.pdf)

Four variation of CNN models are implemented in the code defined as

* CNN-rand: Our baseline model where all
words are randomly initialized and then modified during training.
* CNN-static: A model with pre-trained
vectors from word2vec. All words including the unknown ones that are randomly initialized are kept static and only
the other parameters of the model are learned.
* CNN-non-static: Same as above but the pretrained vectors are fine-tuned for each task.
* CNN-multichannel: A model with two sets
of word vectors. Each set of vectors is treated
as a ‘channel’ and each filter is applied

## Summary Statistics of Dataset

* Dataset Statistics

|   Dataset Name | No. of Classes | Average Length of Sentences |    Max Length of Sentence  |   Dataset Size   |  Number of Word | Number of Words in Word2Vec	| Test Data Size |
|--------|---|----|-------|-------|-------|-------|------|
|   MR   | 2 | 20 |  56   | 10662 | 18765 | 16448 |  CV  |
|  SST1  | 5 | 18 |  53   | 11855 | 17836 | 16262 | 2210 |
|  SST2  | 2 | 18 |  53   | 9613  | 16185 | 14838 | 1821 |
|  Subj  | 2 | 23 |  120  | 10000 | 21323 | 17913 |  CV  |
|  TREC  | 6 | 11 |  38   | 5952  | 9592  | 9125  | 500  |
|   CR   | 2 | 16 |  105  | 3775  | 5340  | 5046  |  CV  |
|  MPQA  | 2 |  3 |  36   | 10606 | 6246  | 6083  |  CV  |

* Dataset Statistics Original Paper

|   Data | c | l  |    N  |   V   |  Vpre | Test |
|--------|---|----|-------|-------|-------|------|
|   MR   | 2 | 20 | 10662 | 18765 | 16448 |  CV  |
|  SST1  | 5 | 18 | 11855 | 17836 | 16262 | 2210 |
|  SST2  | 2 | 19 | 9613  | 16185 | 14838 | 1821 |
|  Subj  | 2 | 23 | 10000 | 21323 | 17913 |  CV  |
|  TREC  | 6 | 10 | 5952  | 9592  | 9125  | 500  |
|   CR   | 2 | 19 | 3775  | 5340  | 5046  |  CV  |
|  MPQA  | 2 |  3 | 10606 | 6246  | 6083  |  CV  |

## Training Results
![Training Results](/results/Results.png)

![Training Results Original Paper](/results/Results Original.png)


## Getting Started

* Training **CNN-rand** model on MR dataset
```
python train.py -dataset=MR -dataset_path=<path to dataset> -keep_embedding_static=False -use_pretrained_vector=False -use_multi_channel=False
```

* Training **CNN-static** model on MR dataset
```
python train.py -dataset=MR -dataset_path=<path to dataset> -keep_embedding_static=True -use_pretrained_vector=True -use_multi_channel=False
```

* Training **CNN-non-static** model on MR dataset
```
python train.py -dataset=MR -dataset_path=<path to dataset> -keep_embedding_static=False -use_pretrained_vector=True -use_multi_channel=False
```

* Training **CNN-multichannel** model on MR dataset
```
python train.py -dataset=MR -dataset_path=<path to dataset> -keep_embedding_static=False -use_pretrained_vector=True -use_multi_channel=True
```

To log the results use argument **log_results**. Model results are stored in folder results.

To save the model use argument **save_model** along with **model_name**. Trained model is stored in models folder.

*Note: Please check parameters section for complete details.*

## Prerequisites


torch==1.4.0

sklearn==0.22.2

## Parameters

```bash
python train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  -dataset DATASET      Name of the Dataset to use
  -dataset_path DATASET_PATH
                        Location to the Dataset
  -epochs EPOCHS        Number of Epochs to train
  -epochs_log EPOCHS_LOG
                        Log Accuracy after every X Epochs
  -optimizer OPTIMIZER  Select optimizer from "Adam" or "Adadelta"
  -lr LR                Learning rate to use while training
  -use_pretrained_vector USE_PRETRAINED_VECTOR
                        To use Pretrained vector (word2vec) or not
  -embedding_size EMBEDDING_SIZE
                        Embedding size to be used in case of self trained
                        embedding only, redundant is using pretrained vector
  -keep_embedding_static KEEP_EMBEDDING_STATIC
                        Would like to train/adjust the Embedding or not
  -model_name MODEL_NAME
                        Provide a name to the model, use the names in the
                        paper for logging
  -save_model SAVE_MODEL
                        Would like to store the model or not, model is stored
                        by dataset name and model name arguments provided
  -batch_size BATCH_SIZE
                        Batch Size to use while training
  -use_multi_channel USE_MULTI_CHANNEL
                        Use multichannel or not
  -log_results LOG_RESULTS
                        Would like to log the final results of the model
```
