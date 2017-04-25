import sys
import glob
import shutil

import tensorflow as tf
tf.set_random_seed(0)

sys.path.append('../common/')

from lstm_common import *
from docopt import docopt
from itertools import count
from evaluation_common import *
from collections import defaultdict
from knowledge_resource import KnowledgeResource
from paths_lstm_classifier_tf import PathLSTMClassifier

EMBEDDINGS_DIM = 50


def main():

    args = docopt("""The LSTM-based integrated pattern-based and distributional method for multiclass
    semantic relations classification

    Usage:
        parse_wikipedia.py <corpus_prefix> <dataset_prefix> <model_prefix_file> <embeddings_file> <num_hidden_layers>

        <wiki_file> = the Wikipedia dump file
        <vocabulary_file> = a file containing the words to include
        <out_file> = the output file
    """)

    corpus_prefix = args['<corpus_prefix>']
    dataset_prefix = args['<dataset_prefix>']
    model_prefix_file = args['<model_prefix_file>']
    embeddings_file = args['<embeddings_file>']
    num_hidden_layers = int(args['<num_hidden_layers>'])

    np.random.seed(133)

    # Load the relations
    with codecs.open(dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = { relation : i for i, relation in enumerate(relations) }

    # Load the datasets
    print 'Loading the dataset...'
    train_set = load_dataset(dataset_prefix + '/train.tsv', relations)
    val_set = load_dataset(dataset_prefix + '/val.tsv', relations)
    test_set = load_dataset(dataset_prefix + '/test.tsv', relations)
    y_train = [relation_index[label] for label in train_set.values()]
    y_val = [relation_index[label] for label in val_set.values()]
    y_test = [relation_index[label] for label in test_set.values()]
    dataset_keys = train_set.keys() + val_set.keys() + test_set.keys()
    print 'Done!'

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(corpus_prefix)
    print 'Done!'

    # Get the vocabulary
    vocabulary = get_vocabulary(corpus, dataset_keys)

    # Load the word embeddings
    print 'Initializing word embeddings...'
    word_vectors, word_index = load_embeddings(embeddings_file, vocabulary)

    # Load the paths and create the feature vectors
    print 'Loading path files...'
    x_y_vectors, dataset_instances, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, \
    dir_inverted_index = load_paths_and_word_vectors(corpus, dataset_keys, word_index)
    print 'Number of words %d, number of pos tags: %d, number of dependency labels: %d, number of directions: %d' % \
          (len(word_index), len(pos_index), len(dep_index), len(dir_index))

    X_train = dataset_instances[:len(train_set)]
    X_val = dataset_instances[len(train_set):len(train_set)+len(val_set)]
    X_test = dataset_instances[len(train_set)+len(val_set):]

    x_y_vectors_train = x_y_vectors[:len(train_set)]
    x_y_vectors_val = x_y_vectors[len(train_set):len(train_set)+len(val_set)]
    x_y_vectors_test = x_y_vectors[len(train_set)+len(val_set):]

    # Tune the hyper-parameters using the validation set
    epochs = [10, 15, 20]
    word_dropout_rates = [0.0, 0.2, 0.4]
    f1_results = []
    descriptions = []
    model_prefixes = []

    for word_dropout_rate in word_dropout_rates:
        for n_epochs in epochs:

            # Create the classifier
            classifier = PathLSTMClassifier(num_lemmas=len(word_index), num_pos=len(pos_index), num_dep=len(dep_index),
                                            num_directions=len(dir_index), n_epochs=n_epochs, num_relations=len(relations),
                                            lemma_embeddings=word_vectors, dropout=word_dropout_rate,
                                            num_hidden_layers=num_hidden_layers)

            description = 'dropout = %.2f, num epochs = %d' % (word_dropout_rate, n_epochs)
            print 'Training with ' + description + '...'
            classifier.fit(X_train, y_train, x_y_vectors=x_y_vectors_train)

            pred = classifier.predict(X_val, x_y_vectors=x_y_vectors_val)
            precision, recall, f1, support = evaluate(y_val, pred, relations, do_full_reoprt=False)
            print 'Dropout = %f, num epochs = %d, Precision: %.3f, Recall: %.3f, F1: %.3f' % \
                  (word_dropout_rate, n_epochs, precision, recall, f1)
            f1_results.append(f1)

            # Save intermediate models
            curr_model_prefix = '%s_%.2f_%d' % (model_prefix_file, word_dropout_rate, n_epochs)
            model_prefixes.append(curr_model_prefix)
            classifier.save_model(curr_model_prefix, [word_index, pos_index, dep_index, dir_index])
            descriptions.append(description)
            classifier.close()

    best_index = np.argmax(f1_results)
    description = descriptions[best_index]
    print 'Best hyper-parameters: ' + description

    # Save the best model to a file
    print 'Saving the model...'
    best_model_prefix = model_prefixes[best_index]
    for file in glob.glob(best_model_prefix + '.*'):
        shutil.copy(file, model_prefix_file + file[file.index(best_model_prefix) + len(best_model_prefix):])

    classifier, word_index, pos_index, dep_index, dir_index = PathLSTMClassifier.load_model(model_prefix_file)

    # Evaluate on the test set
    print 'Evaluation:'
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
    precision, recall, f1, support = evaluate(y_test, pred, relations, do_full_reoprt=True)
    print 'Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1)
    classifier.close()

    # Write the predictions to a file
    output_predictions(model_prefix_file + '.predictions', relations, pred, test_set.keys(), y_test)


def get_vocabulary(corpus, dataset_keys):
    """
    Get all the words in the dataset and paths
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :return: a set of distinct words appearing as x or y or in a path
    """
    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    path_lemmas = set([edge.split('/')[0]
                       for (x_id, y_id) in keys
                       for path in get_paths(corpus, x_id, y_id).keys()
                       for edge in path.split('_')])
    x_y_words = set([x for (x, y) in dataset_keys]).union([y for (x, y) in dataset_keys])
    return list(path_lemmas.union(x_y_words))


def load_paths_and_word_vectors(corpus, dataset_keys, lemma_index):
    """
    Load the paths and the word vectors for this dataset
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :param word_index: the index of words for the word embeddings
    :return:
    """

    # Define the dictionaries
    pos_index = defaultdict(count(0).next)
    dep_index = defaultdict(count(0).next)
    dir_index = defaultdict(count(0).next)

    dummy = pos_index['#UNKNOWN#']
    dummy = dep_index['#UNKNOWN#']
    dummy = dir_index['#UNKNOWN#']

    # Vectorize tha paths
    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    paths_x_to_y = [{ vectorize_path(path, lemma_index, pos_index, dep_index, dir_index) : count
                      for path, count in get_paths(corpus, x_id, y_id).iteritems() }
                    for (x_id, y_id) in keys]
    paths = [ { p : c for p, c in paths_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(path_list.keys()) == 0]
    print 'Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys)

    # Get the word embeddings for x and y (get a lemma index)
    print 'Getting word vectors for the terms...'
    x_y_vectors = [(lemma_index.get(x, 0), lemma_index.get(y, 0)) for (x, y) in dataset_keys]

    pos_inverted_index = { i : p for p, i in pos_index.iteritems() }
    dep_inverted_index = { i : p for p, i in dep_index.iteritems() }
    dir_inverted_index = { i : p for p, i in dir_index.iteritems() }

    print 'Done loading corpus data!'

    return x_y_vectors, paths, pos_index, dep_index, dir_index, pos_inverted_index, dep_inverted_index, \
           dir_inverted_index


if __name__ == '__main__':
    main()
