import bsddb
import codecs

from docopt import docopt
from collections import defaultdict


def main():
    """
    Creates a "knowledge resource" from triplets file
    """

    # Get the arguments
    args = docopt("""Creates a knowledge resource from triplets file.

    Usage:
        create_resource_from_corpus.py <triplets_file> <resource_prefix>

        <triplets_file> = the file that contains text triplets, formatted as X\tY\tpath, obtained by parse_wikipedia.py.
        <resource_prefix> = the file names' prefix for the resource files. This directory should contain the entities
        file (resource_prefix + 'Entities.txt') and the path file (resource_prefix + 'Paths.txt'),
        containing the relevant entities and paths, each in a separate line.
        The entities file is the vocabulary: in our experiments, we used the common 100k bigrams and
        trigrams in Wikipedia + the GloVe vocabulary (400k). The path file contains the most
        frequent paths in Wikipedia (as described in https://github.com/vered1986/HypeNET).
    """)
    triplets_file = args['<triplets_file>']
    resource_prefix = args['<resource_prefix>']

    # Load the terms and create the dictionary
    term_to_id, id_to_term = load_map(resource_prefix + 'Entities.txt')
    term_to_id_db = bsddb.btopen(resource_prefix + '_term_to_id.db', 'c')
    id_to_term_db = bsddb.btopen(resource_prefix + '_id_to_term.db', 'c')

    for id, term in id_to_term.iteritems():
        id, term = str(id), str(term)
        term_to_id_db[term] = id
        id_to_term_db[id] = term

    # Load the paths and create the dictionary
    path_to_id, id_to_path = load_map(resource_prefix + 'Paths.txt')
    path_to_id_db = bsddb.btopen(resource_prefix + '_path_to_id.db', 'c')
    id_to_path_db = bsddb.btopen(resource_prefix + '_id_to_path.db', 'c')

    for id, path in id_to_path.iteritems():
        id, path = str(id), str(path)
        path_to_id_db[path] = id
        id_to_path_db[id] = path

    path_to_id_db.sync()
    id_to_path_db.sync()

    # Load the triplets file
    l2r_edges = defaultdict(lambda : defaultdict(lambda : defaultdict(int)))

    num_line = 0

    with codecs.open(triplets_file) as f_in:
        for line in f_in:
            try:
                x, y, path = line.strip().split('\t')
            except:
                print line
                continue

            x_id, y_id, path_id = term_to_id.get(x, -1), term_to_id.get(y, -1), path_to_id.get(path, -1)
            if x_id > -1 and y_id > -1 and path_id > -1:
                l2r_edges[x_id][y_id][path_id] += 1

            num_line += 1
            if num_line % 1000000 == 0:
                print '#1: processed ', num_line, ' lines.'

    with codecs.open(resource_prefix + '-l2r.txt', 'w', 'utf-8') as f_out:
        for x in l2r_edges.keys():
            for y in l2r_edges[x].keys():
                print >> f_out, str(x) + '\t' + str(y) + '\t' + ','.join(
                    [':'.join((str(p), str(val))) for (p, val) in l2r_edges[x][y].iteritems()])

    # Convert to dictionary
    l2r_db = bsddb.btopen(resource_prefix + '_l2r.db', 'c')

    num_line = 0

    with open(resource_prefix + '-l2r.txt') as f_in:
        for line in f_in:
            x, y, path_str = line.strip().split('\t')
            l2r_db[x + '###' + y] = path_str

            num_line += 1
            if num_line % 1000000 == 0:
                print '#2: processed ', num_line, ' lines.'

    l2r_db.sync()


def load_map(str_file):
    """
    Loads the map of term/property string to ID
    :param str_file - a file containing a list of strings
    """
    with codecs.open(str_file, 'r', 'utf-8') as f_in:
        lines = [line.strip() for line in f_in]
        id_to_str = { i : s for i, s in enumerate(lines) }
        str_to_id = { s : i for i, s in id_to_str.iteritems() }

    return str_to_id, id_to_str


if __name__ == '__main__':
    main()
