import math
import json
from __main__ import args

# Support GPU, following change https://github.com/vered1986/LexNET/pull/2 from @gossebouma
# Use CPU
if args.gpus == 0:
    import _dynet as dy
    dyparams = dy.DynetParams()

# Use GPU
else:
    import _gdynet as dy
    dyparams = dy.DynetParams()
    dyparams.set_requested_gpus(args.gpus)

dyparams.set_mem(args.memory)
dyparams.set_random_seed(args.seed)
dyparams.init()
	
from lstm_common import *
from sklearn.base import BaseEstimator

NUM_LAYERS = 2
LSTM_HIDDEN_DIM = 60
LEMMA_DIM = 50
POS_DIM = 4
DEP_DIM = 5
DIR_DIM = 1

EMPTY_PATH = ((0, 0, 0, 0),)
LOSS_EPSILON = 0.0 # 0.01
MINIBATCH_SIZE = 100


class PathLSTMClassifier(BaseEstimator):

    def __init__(self, num_lemmas, num_pos, num_dep, num_directions=5, n_epochs=10, num_relations=2,
                 alpha=0.01, lemma_embeddings=None, dropout=0.0, use_xy_embeddings=False, num_hidden_layers=0):
        """'
        Initialize the LSTM
        :param num_lemmas Number of distinct lemmas
        :param num_pos Number of distinct part of speech tags
        :param num_dep Number of distinct depenedency labels
        :param num_directions Number of distinct path directions (e.g. >,<)
        :param n_epochs Number of training epochs
        :param num_relations Number of classes (e.g. binary = 2)
        :param alpha Learning rate
        :param lemma_embeddings Pre-trained word embedding vectors for the path-based component
        :param dropout Dropout rate
        :param use_xy_embeddings Whether to concatenate x and y word embeddings to the network input
        :param num_hidden_layers The number of hidden layers for the term-pair classification network
        """
        self.n_epochs = n_epochs
        self.num_lemmas = num_lemmas
        self.num_pos = num_pos
        self.num_dep = num_dep
        self.num_directions = num_directions
        self.num_relations = num_relations
        self.alpha = alpha
        self.dropout = dropout
        self.use_xy_embeddings = use_xy_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.update = True

        self.lemma_vectors = None
        if lemma_embeddings is not None:
            self.lemma_vectors = lemma_embeddings
            self.lemma_embeddings_dim = lemma_embeddings.shape[1]
        else:
            self.lemma_embeddings_dim = LEMMA_DIM

        # Create the network
        print 'Creating the network...'
        self.builder, self.model, self.model_parameters = create_computation_graph(self.num_lemmas, self.num_pos,
                                                                                   self.num_dep, self.num_directions,
                                                                                   self.num_relations, self.lemma_vectors,
                                                                                   use_xy_embeddings, self.num_hidden_layers,
                                                                                   self.lemma_embeddings_dim)
        print 'Done!'

    def fit(self, X_train, y_train, x_y_vectors=None):
        """
        Train the model
        """
        print 'Training the model...'
        train(self.builder, self.model, self.model_parameters, X_train, y_train, self.n_epochs, self.alpha, self.update,
              self.dropout, x_y_vectors, self.num_hidden_layers)
        print 'Done!'

    def save_model(self, output_prefix, dictionaries):
        """
        Save the trained model to a file
        """
        self.model.save(output_prefix + '.model')

        # Save the model parameter shapes
        lookups = ['lemma_lookup', 'pos_lookup', 'dep_lookup', 'dir_lookup']
        params = { param_name : self.model_parameters[param_name].shape() for param_name in lookups }
        params['num_relations'] = self.num_relations
        params['use_xy_embeddings'] = self.use_xy_embeddings
        params['num_hidden_layers'] = self.num_hidden_layers

        with open(output_prefix + '.params', 'w') as f_out:
            json.dump(params, f_out, indent=2)

        # Save the dictionaries
        with open(output_prefix + '.dict', 'w') as f_out:
            json.dump(dictionaries, f_out, indent=2)

    def predict(self, X_test, x_y_vectors=None):
        """
        Predict the classification of the test set
        """
        model = self.model
        model_parameters = self.model_parameters
        builder = self.builder
        test_pred = []

        # Predict every 100 instances together
        for chunk in xrange(0, len(X_test), MINIBATCH_SIZE):
            dy.renew_cg()
            path_cache = {}
            test_pred.extend([np.argmax(process_one_instance(
                builder, model, model_parameters, path_set, path_cache, self.update, dropout=0.0,
                x_y_vectors=x_y_vectors[chunk + i] if x_y_vectors is not None else None,
                num_hidden_layers=self.num_hidden_layers).npvalue())
                              for i, path_set in enumerate(X_test[chunk:chunk+MINIBATCH_SIZE])])

        return test_pred

    def predict_with_score(self, X_test, x_y_vectors=None):
        """
        Predict the classification of the test set
        """
        model = self.model
        builder = self.builder

        dy.renew_cg()

        path_cache = {}
        test_pred = [process_one_instance(builder, model, path_set, path_cache, self.update, dropout=0.0,
                                          x_y_vectors=x_y_vectors[i] if x_y_vectors is not None else None,
                                          num_hidden_layers=self.num_hidden_layers).npvalue()
                     for i, path_set in enumerate(X_test)]

        return [(np.argmax(vec), vec[np.argmax(vec)]) for vec in test_pred]

    def get_top_k_paths(self, all_paths, relation_index, threshold):
        """
        Get the top k scoring paths
        """
        builder = self.builder
        model = self.model
        model_parameters = self.model_parameters
        lemma_lookup = model_parameters['lemma_lookup']
        pos_lookup = model_parameters['pos_lookup']
        dep_lookup = model_parameters['dep_lookup']
        dir_lookup = model_parameters['dir_lookup']

        path_scores = []

        for i, path in enumerate(all_paths):

            if i % 1000 == 0:
                cg = dy.renew_cg()
                W1 = dy.parameter(model_parameters['W1'])
                b1 = dy.parameter(model_parameters['b1'])
                W2 = None
                b2 = None

                if self.num_hidden_layers == 1:
                    W2 = dy.parameter(model_parameters['W2'])
                    b2 = dy.parameter(model_parameters['b2'])

            path_embedding = get_path_embedding(builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path)

            if self.use_xy_embeddings:
                zero_word = dy.inputVector([0.0] * self.lemma_embeddings_dim)
                path_embedding = dy.concatenate([zero_word, path_embedding, zero_word])

            h = W1 * path_embedding + b1

            if self.num_hidden_layers == 1:
                h = W2 * dy.tanh(h) + b2

            path_score = dy.softmax(h).npvalue().T
            path_scores.append(path_score)

        path_scores = np.vstack(path_scores)

        top_paths = []
        for i in range(len(relation_index)):
            indices = np.argsort(-path_scores[:, i])
            top_paths.append([(all_paths[index], path_scores[index, i]) for index in indices
                         if threshold is None or path_scores[index, i] >= threshold])

        return top_paths


def process_one_instance(builder, model, model_parameters, instance, path_cache, update=True, dropout=0.0,
                         x_y_vectors=None, num_hidden_layers=0):
    """
    Return the LSTM output vector of a single term-pair - the average path embedding
    :param builder: the LSTM builder
    :param model: the LSTM model
    :param model_parameters: the model parameters
    :param instance: a Counter object with paths
    :param path_cache: the cache for path embeddings
    :param update: whether to update the lemma embeddings
    :param dropout: word dropout rate
    :param x_y_vectors: the current word vectors for x and y
    :param num_hidden_layers The number of hidden layers for the term-pair classification network
    :return: the LSTM output vector of a single term-pair
    """
    W1 = dy.parameter(model_parameters['W1'])
    b1 = dy.parameter(model_parameters['b1'])
    W2 = None
    b2 = None

    if num_hidden_layers == 1:
        W2 = dy.parameter(model_parameters['W2'])
        b2 = dy.parameter(model_parameters['b2'])

    lemma_lookup = model_parameters['lemma_lookup']
    pos_lookup = model_parameters['pos_lookup']
    dep_lookup = model_parameters['dep_lookup']
    dir_lookup = model_parameters['dir_lookup']

    # Use the LSTM output vector and feed it to the MLP

    # Add the empty path
    paths = instance

    if len(paths) == 0:
        paths[EMPTY_PATH] = 1

    # Compute the averaged path
    num_paths = reduce(lambda x, y: x + y, instance.itervalues())
    path_embbedings = [get_path_embedding_from_cache(path_cache, builder, lemma_lookup, pos_lookup, dep_lookup,
                                                     dir_lookup, path, update, dropout) * count
                       for path, count in instance.iteritems()]
    input_vec = dy.esum(path_embbedings) * (1.0 / num_paths)

    # Concatenate x and y embeddings
    if x_y_vectors is not None:
        x_vector, y_vector = dy.lookup(lemma_lookup, x_y_vectors[0]), dy.lookup(lemma_lookup, x_y_vectors[1])
        input_vec = dy.concatenate([x_vector, input_vec, y_vector])

    h = W1 * input_vec + b1

    if num_hidden_layers == 1:
        h = W2 * dy.tanh(h) + b2

    output = dy.softmax(h)

    return output


def get_path_embedding_from_cache(cache, builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path,
                                  update=True, dropout=0.0):
    """

    :param cache: the cache for the path embeddings
    :param builder: the LSTM builder
    :param lemma_lookup: the lemma embeddings lookup table
    :param pos_lookup: the part-of-speech embeddings lookup table
    :param dep_lookup: the dependency label embeddings lookup table
    :param dir_lookup: the direction embeddings lookup table
    :param path: sequence of edges
    :param update: whether to update the lemma embeddings
    :param dropout: the word drop out rate
    :return: the path embedding, computed by the LSTM or retrieved from the cache
    """

    if path not in cache:
        cache[path] = get_path_embedding(builder, lemma_lookup, pos_lookup, dep_lookup,
                                         dir_lookup, path, update, dropout)
    return cache[path]


def get_path_embedding(builder, lemma_lookup, pos_lookup, dep_lookup, dir_lookup, path, update=True, drop=0.0):
    """
    Get a vector representing a path
    :param builder: the LSTM builder
    :param lemma_lookup: the lemma embeddings lookup table
    :param pos_lookup: the part-of-speech embeddings lookup table
    :param dep_lookup: the dependency label embeddings lookup table
    :param dir_lookup: the direction embeddings lookup table
    :param path: sequence of edges
    :param update: whether to update the lemma embeddings
    :return: a vector representing a path
    """

    # Concatenate the edge components to one vector
    inputs = [dy.concatenate([word_dropout(lemma_lookup, edge[0], drop, update),
                           word_dropout(pos_lookup, edge[1], drop),
                           word_dropout(dep_lookup, edge[2], drop),
                           word_dropout(dir_lookup, edge[3], drop)])
                 for edge in path]

    return builder.initial_state().transduce(inputs)[-1]


def word_dropout(lookup_table, word, rate, update=True):
    """
    Apply word dropout with dropout rate
    :param exp: expression vector
    :param rate: dropout rate
    :return:
    """
    new_word = np.random.choice([word, 0], size=1, p=[1 - rate, rate])[0]
    return dy.lookup(lookup_table, new_word, update)


def train(builder, model, model_parameters, X_train, y_train, nepochs, alpha=0.01, update=True, dropout=0.0,
          x_y_vectors=None, num_hidden_layers=0):
    """
    Train the LSTM
    :param builder: the LSTM builder
    :param model: LSTM RNN model
    :param model_parameters: the model parameters
    :param X_train: the train instances
    :param y_train: the train labels
    :param nepochs: number of epochs
    :param alpha: the learning rate (only for SGD)
    :param update: whether to update the lemma embeddings
    :param dropout: dropout probability for all component embeddings
    :param x_y_vectors: the word vectors of x and y
    :param num_hidden_layers The number of hidden layers for the term-pair classification network
    """
    trainer = dy.AdamTrainer(model, alpha=alpha)
    minibatch_size = min(MINIBATCH_SIZE, len(y_train))
    nminibatches = int(math.ceil(len(y_train) / minibatch_size))
    previous_loss = 1000

    for epoch in range(nepochs):

        total_loss = 0.0

        epoch_indices = np.random.permutation(len(y_train))

        for minibatch in range(nminibatches):

            path_cache = {}
            batch_indices = epoch_indices[minibatch * minibatch_size:(minibatch + 1) * minibatch_size]

            dy.renew_cg()

            loss = dy.esum([-dy.log(dy.pick(
                process_one_instance(builder, model, model_parameters, X_train[batch_indices[i]], path_cache, update,
                                     dropout, x_y_vectors=x_y_vectors[batch_indices[i]] if x_y_vectors is not None else None,
                                     num_hidden_layers=num_hidden_layers),
                y_train[batch_indices[i]])) for i in range(minibatch_size)])
            total_loss += loss.value() # forward computation
            loss.backward()
            trainer.update()

        # deprecated http://dynet.readthedocs.io/en/latest/python_ref.html#optimizers GB
        # and requires an argument (would be epoch i guess...)
        # trainer.update_epoch()
        trainer.update()
        total_loss /= len(y_train)
        print 'Epoch', (epoch + 1), '/', nepochs, 'Loss =', total_loss

        # Early stopping
        if math.fabs(previous_loss - total_loss) < LOSS_EPSILON:
            break

        previous_loss = total_loss


def create_computation_graph(num_lemmas, num_pos, num_dep, num_directions, num_relations,
                             wv=None, use_xy_embeddings=False, num_hidden_layers=0, lemma_dimension=50):
    """
    Initialize the model
    :param num_lemmas Number of distinct lemmas
    :param num_pos Number of distinct part of speech tags
    :param num_dep Number of distinct depenedency labels
    :param num_directions Number of distinct path directions (e.g. >,<)
    :param num_relations Number of classes (e.g. binary = 2)
    :param wv Pre-trained word embeddings file
    :param use_xy_embeddings Whether to concatenate x and y word embeddings to the network input
    :param num_hidden_layers The number of hidden layers for the term-pair classification network
    :param lemma_dimension The dimension of the lemma embeddings
    :return:
    """
    # model = Model() -- gives error? tried to fix by looking at dynet tutorial examples -- GB
    dy.renew_cg()
    model = dy.ParameterCollection()
    network_input = LSTM_HIDDEN_DIM

    builder = dy.LSTMBuilder(NUM_LAYERS, lemma_dimension + POS_DIM + DEP_DIM + DIR_DIM, network_input, model)

    # Concatenate x and y
    if use_xy_embeddings:
        network_input += 2 * lemma_dimension

    #  'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'
    hidden_dim = int((network_input + num_relations) / 2)

    model_parameters = {}

    if num_hidden_layers == 0:
        model_parameters['W1'] = model.add_parameters((num_relations, network_input))
        model_parameters['b1'] = model.add_parameters((num_relations, 1))

    elif num_hidden_layers == 1:

        model_parameters['W1'] = model.add_parameters((hidden_dim, network_input))
        model_parameters['b1'] = model.add_parameters((hidden_dim, 1))
        model_parameters['W2'] = model.add_parameters((num_relations, hidden_dim))
        model_parameters['b2'] = model.add_parameters((num_relations, 1))

    else:
        raise ValueError('Only 0 or 1 hidden layers are supported')

    model_parameters['lemma_lookup'] = model.add_lookup_parameters((num_lemmas, lemma_dimension))

    # Pre-trained word embeddings
    if wv is not None:
        model_parameters['lemma_lookup'].init_from_array(wv)

    model_parameters['pos_lookup'] = model.add_lookup_parameters((num_pos, POS_DIM))
    model_parameters['dep_lookup'] = model.add_lookup_parameters((num_dep, DEP_DIM))
    model_parameters['dir_lookup'] = model.add_lookup_parameters((num_directions, DIR_DIM))

    return builder, model, model_parameters


def load_model(model_file_prefix):
    """
    Load the trained model from a file
    """

    # Load the parameters from the json file
    with open(model_file_prefix + '.params') as f_in:
        params = json.load(f_in)

    classifier = PathLSTMClassifier(params['lemma_lookup'][0], params['pos_lookup'][0], params['dep_lookup'][0],
                                    params['dir_lookup'][0], num_relations=params['num_relations'],
                                    use_xy_embeddings=params['use_xy_embeddings'],
                                    num_hidden_layers=params['num_hidden_layers'])

    # Load the model
    classifier.model.populate(model_file_prefix + '.model')

    # Load the dictionaries from the json file
    with open(model_file_prefix + '.dict') as f_in:
        dictionaries = json.load(f_in)

    word_index, pos_index, dep_index, dir_index = dictionaries

    return classifier, word_index, pos_index, dep_index, dir_index