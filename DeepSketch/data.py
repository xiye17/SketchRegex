import random
from utils import *
from os.path import join

PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"

# Wrapper class for an example.
# x = the natural language as one string
# x_tok = tokenized NL, a list of strings
# x_indexed = indexed tokens, a list of ints
# y = the logical form
# y_tok = tokenized logical form, a list of strings
# y_indexed = indexed logical form
class Example(object):
    def __init__(self, x, x_tok, x_indexed, y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed
        self.id = 0

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + "\n   indexed as: " + repr(self.x_indexed) + " => " + repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()

# Wrapper for a Derivation consisting of an Example object, a score/probability associated with that example,
# and the tokenized prediction.
class Derivation(object):
    def __init__(self, example, p, y_toks):
        self.example = example
        self.p = p
        self.y_toks = y_toks

    def __str__(self):
        return "%s (%s)" % (self.y_toks, self.p)

    def __repr__(self):
        return self.__str__()

def read_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip('\n') for x in lines]

    return lines

def get_cache_file(cache_id):
    return join('./caches', cache_id + '.pkl')

def get_model_file(dataset, model_id):
    return join('./checkpoints', dataset, model_id + '.tar')

# Reads the training, dev, and test data from the corresponding files.
def load_datasets(dataset):
    output_path = join('./datasets', dataset)
    
    train_raw = load_dataset(output_path, 'train')
    dev_raw = load_dataset(output_path, 'val')
    
    input_indexer = Indexer()
    output_indexer = Indexer()

    input_indexer.load_from_file(join(output_path, 'src-indexer.pkl'))
    output_indexer.load_from_file(join(output_path, 'targ-indexer.pkl'))
    return train_raw, dev_raw, input_indexer, output_indexer

def load_test_dataset(dataset, split):
    output_path = join('./datasets', dataset)
    
    test_raw = load_dataset(output_path, split)
    
    input_indexer = Indexer()
    output_indexer = Indexer()

    input_indexer.load_from_file(join(output_path, 'src-indexer.pkl'))
    output_indexer.load_from_file(join(output_path, 'targ-indexer.pkl'))
    return test_raw, input_indexer, output_indexer


# Reads a dataset in from the given file
def load_dataset(output_path, split):

    src_lines = read_lines(join(output_path, 'src-%s.txt' % (split)))
    targ_lines = read_lines(join(output_path, 'targ-%s.txt' % (split)))

    data_raw = list(zip(src_lines, targ_lines))

    return data_raw


# Whitespace tokenization
def tokenize(x):
    return x.split()


def index(x_tok, indexer):
    return [indexer.index_of(xi) if indexer.index_of(xi) >= 0 else indexer.index_of(UNK_SYMBOL) for xi in x_tok]


def index_data(data, input_indexer, output_indexer, example_len_limit):
    data_indexed = []
    for (x, y) in data:
        x_tok = tokenize(x)
        y_tok = tokenize(y)[0:example_len_limit]
        data_indexed.append(Example(x, x_tok, index(x_tok, input_indexer), y, y_tok,
                                          index(y_tok, output_indexer) + [output_indexer.get_index(EOS_SYMBOL)]))
    for i, exs in enumerate(data_indexed):
        exs.id = i + 1
    return data_indexed

def filter_data(data_indexed):
    return [exs for exs in data_indexed if not (exs.y in ["null", 'empty', 'none'])]

def tricky_filter_data(data_indexed):
    test_kb, input_indexer, output_indexer = load_test_dataset("KB13Plus", "val")
    test_kb = index_data(test_kb, input_indexer, output_indexer, 65)
    return [x[0] for x in zip(data_indexed, test_kb) if x[1].y != "null"]

# Indexes train and test datasets where all words occurring less than or equal to unk_threshold times are
# replaced by UNK tokens.
def index_datasets(train_data, dev_data, input_indexer, output_indexer, example_len_limit):
    # Index things
    train_data_indexed = index_data(train_data, input_indexer, output_indexer, example_len_limit)
    dev_data_indexed = index_data(dev_data, input_indexer, output_indexer, example_len_limit)
    # test_data_indexed = index_data(test_data, input_indexer, output_indexer, example_len_limit)
    return train_data_indexed, dev_data_indexed

# Geoquery preprocessing adapted from Jia and Liang. Standardizes variable names with De Brujin indices -- just a
# smarter way of indexing variables in statements to make parsing easier.
def geoquery_preprocess_lf(lf):
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
        if w.isalpha() and len(w) == 1:
            if w in cur_vars:
                ind_from_end = len(cur_vars) - cur_vars.index(w) - 1
                new_toks.append('V%d' % ind_from_end)
            else:
                cur_vars.append(w)
                new_toks.append('NV')
        else:
            new_toks.append(w)
    return ' '.join(new_toks)
