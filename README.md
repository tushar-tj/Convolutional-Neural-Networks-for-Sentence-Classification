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
![Dataset Statistics]()

![Dataset Statistics Original Paper]()

## Training Results
![Training Results]()

![Training Results Original Paper]()


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
./train.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  --embedding_dim EMBEDDING_DIM
                        Dimensionality of character embedding (default: 128)
  --filter_sizes FILTER_SIZES
                        Comma-separated filter sizes (default: '3,4,5')
  --num_filters NUM_FILTERS
                        Number of filters per filter size (default: 128)
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularizaion lambda (default: 0.0)
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability (default: 0.5)
  --batch_size BATCH_SIZE
                        Batch Size (default: 64)
  --num_epochs NUM_EPOCHS
                        Number of training epochs (default: 100)
  --evaluate_every EVALUATE_EVERY
                        Evaluate model on dev set after this many steps
                        (default: 100)
  --checkpoint_every CHECKPOINT_EVERY
                        Save model after this many steps (default: 100)
  --allow_soft_placement ALLOW_SOFT_PLACEMENT
                        Allow device soft device placement
  --noallow_soft_placement
  --log_device_placement LOG_DEVICE_PLACEMENT
                        Log placement of ops on devices
  --nolog_device_placement
```