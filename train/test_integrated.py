import sys
sys.argv.insert(1, '--cnn-mem')
sys.argv.insert(2, '8192')
sys.argv.insert(3, '--cnn-seed')
sys.argv.insert(4, '3016748844')
sys.path.append('../common/')

sys.path.append('../common')

from evaluation_common import *
from paths_lstm_classifier import *
from knowledge_resource import KnowledgeResource

EMBEDDINGS_DIM = 50


def main():

    # The LSTM-based integrated pattern-based and distributional method for multiclass semantic relations classification
    corpus_prefix = sys.argv[5]
    dataset_prefix = sys.argv[6]
    model_file_prefix = sys.argv[7]

    # Load the relations
    with codecs.open(dataset_prefix + '/relations.txt', 'r', 'utf-8') as f_in:
        relations = [line.strip() for line in f_in]
        relation_index = { relation : i for i, relation in enumerate(relations) }

    # Load the datasets
    print 'Loading the dataset...'
    test_set = load_dataset(dataset_prefix + '/test.tsv', relations)
    y_test = [relation_index[label] for label in test_set.values()]

    # Load the resource (processed corpus)
    print 'Loading the corpus...'
    corpus = KnowledgeResource(corpus_prefix)
    print 'Done!'

    # Load the pre-trained model file
    classifier, word_index, pos_index, dep_index, dir_index = load_model(model_file_prefix)

    # Load the paths and create the feature vectors
    print 'Loading path files...'
    x_y_vectors_test, X_test = load_paths_and_word_vectors(corpus, test_set.keys(),
                                                           word_index, pos_index, dep_index, dir_index)

    print 'Evaluation:'
    pred = classifier.predict(X_test, x_y_vectors=x_y_vectors_test)
    precision, recall, f1, support = evaluate(y_test, pred, relations, do_full_reoprt=True)
    print 'Precision: %.3f, Recall: %.3f, F1: %.3f' % (precision, recall, f1)


def load_paths_and_word_vectors(corpus, dataset_keys, word_index, pos_index, dep_index, dir_index):
    """
    Load the paths and the word vectors for this dataset
    :param corpus: the corpus object
    :param dataset_keys: the word pairs in the dataset
    :param word_index: the index of words for the word embeddings
    :return:
    """

    # Vectorize tha paths
    keys = [(corpus.get_id_by_term(str(x)), corpus.get_id_by_term(str(y))) for (x, y) in dataset_keys]
    paths_x_to_y = [{ vectorize_path(path, word_index, pos_index, dep_index, dir_index) : count
                      for path, count in get_paths(corpus, x_id, y_id).iteritems() }
                    for (x_id, y_id) in keys]
    paths = [ { p : c for p, c in paths_x_to_y[i].iteritems() if p is not None } for i in range(len(keys)) ]

    empty = [dataset_keys[i] for i, path_list in enumerate(paths) if len(path_list.keys()) == 0]
    print 'Pairs without paths:', len(empty), ', all dataset:', len(dataset_keys)

    # Get the word embeddings for x and y (get a lemma index)
    print 'Getting word vectors for the terms...'
    x_y_vectors = [(word_index.get(x, 0), word_index.get(y, 0)) for (x, y) in dataset_keys]

    print 'Done loading corpus data!'

    return x_y_vectors, paths


if __name__ == '__main__':
    main()
