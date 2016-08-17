# LexNET: Integrated Path-based and Distributional Method for Lexical Semantic Relation Classification

This is the code used in the paper:

<b>"The Roles of Path-based and Distributional Information in Recognizing Lexical Semantic Relations"</b><br/>
Vered Shwartz and Ido Dagan. (link - TBD)

It is used to classify semantic relations between term-pairs, using disributional information on each term, and path-based information, encoded using an LSTM.

***

<b>Prerequisites:</b>
* Python 2.7
* Numpy
* scikit-learn
* [bsddb](https://docs.python.org/2/library/bsddb.html)
* [PyCNN](https://github.com/clab/cnn/)

<b>Quick Start:</b>

The repository contains the following directories:
* common - the knowledge resource class, which is used by other models to save the path data from the corpus (should be copied to other directories).
* corpus - code for parsing the corpus and extracting paths.
* dataset - the datasets used in the paper.
* train - code for training and testing the LexNET model, and pre-trained models for the datasets (TBD).

To train LexNET, run:

`train_integrated.py [corpus_prefix] [dataset_prefix] [model_prefix_file] [embeddings_file] [num_hidden_layers]`

Where:
* `corpus_prefix` is the file path and prefix of the corpus files, e.g. `corpus/wiki`, such that the directory corpus contains the `wiki_*.db` files created by `create_resource_from_corpus.py`.
* `dataset_prefix` is the file path of the dataset files, e.g. `dataset/combined`, such that this directory contains 3 files: `train.tsv`, `test.tsv` and `val.tsv`.
* `model_prefix_file` is the output directory and prefix for the model files. The model is saved in 3 files: `.model`, `.params` and `.dict.`
In addition, the test set predictions are saved in `.predictions`, and the prominent paths are saved to `.paths`.
* `embeddings_file` is the pre-trained word embeddings file, in txt format (i.e., every line consists of the word, followed by a space, and its vector. See [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) for an example.)
* `num_hidden_layers` is the number of network hidden layers (0 and 1 are supported).

The script trains several models, tuning the word dropout rate and the learning rate using the validation set. The best performing model on the validation set is saved and evaluated on the test set.

<b>Datasets:</b>
The datasets used in this paper are available in the datasets directory, split to train, test and validation sets.

* K&H+N ([Necsulescu et al., 2015](http://www.aclweb.org/anthology/S15-1021))
* BLESS ([Baroni and Lenci, 2011](http://www.aclweb.org/anthology/W11-2501))
* EVALution ([Santus et al., 2015](http://www.aclweb.org/anthology/W15-4208))
* ROOT09 ([Santus et al., 2016](http://arxiv.org/abs/1603.08702))

In addition, we experimented with a combined dataset, which is also available in the directory.

<b>Corpus:</b>
In our paper we use the English Wikipedia dump from May 2015 as the corpus. We computed the paths between the most frequent unigrams, bigrams and trigrams in Wikipedia (based on [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) vocabulary and the most frequent 100k bigrams and trigrams). Rather than re-creating the corpus, if you'd like to use the same corpus, the files for the Wiki corpus are available [here](https://drive.google.com/folderview?id=0B0kBcFEBhcbhdXBTOVRRbThOVDg&usp=sharing).

<b>Pretrained Models:</b>
Since the datasets we used in this work differ from each other, we recommend training the model rather than using pretrained models. If you prefer using our pretrained models, they are available [here](https://drive.google.com/folderview?id=0B0kBcFEBhcbhQ3h6VDV2NFQ0SGc&usp=sharing). (Incomplete)
