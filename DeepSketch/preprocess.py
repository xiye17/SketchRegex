import argparse
import random
import numpy as np
import time
import sys
import argparse
import subprocess
import os
from os.path import join
import errno
import math
import re
from data import *
from utils import *

def _parse_args():
    parser = argparse.ArgumentParser(description='preprocess.py')

    parser.add_argument('dataset', help='specified dataset')
    parser.add_argument('--no_test', dest='with_test', default=True, action='store_false', help='no test split')
    parser.add_argument('--seed', type=int, default=23333, help='RNG seed (default = None)')
    parser.add_argument('--do_index', dest='do_index', default=False, action='store_true')
    parser.add_argument('--do_stats', dest='do_stats', default=False, action='store_true')
    args = parser.parse_args()
    return args

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def process_tokens(data_dir, targ_separate=True):
    desc_file_name = "{}/{}".format(data_dir, "src")
    regex_file_name = "{}/{}".format(data_dir, "targ")

    if targ_separate:
        regex_lines = [" ".join(line.rstrip('\n')) for line in open('{}{}'.format(regex_file_name, '.txt'))]
    else:
        regex_lines = ["".join(line.rstrip('\n')) for line in open('{}{}'.format(regex_file_name, '.txt'))]

    desc_lines = [" " + line.rstrip('\n') + " " for line in open('{}{}'.format(desc_file_name, '.txt'))]

    desc_lines = [line.lower() for line in desc_lines]
    punc = [',', '.', '!', ';']
    for p in punc:
        p_space = '{} '.format(p)
        p_2_space = ' {} '.format(p)
        desc_lines = [line.replace(p_space, p) for line in desc_lines]
        desc_lines = [line.replace(p, p_2_space) for line in desc_lines]


    num_pairs = [(' one ', ' 1 '), (' two ', ' 2 '), (' three ', ' 3 '), (' four ', ' 4 '),
    (' five ', '5'), (' six ', '6'), (' seven ', ' 7 '), (' eight ', ' 8 '), (' nine ', ' 9 '), (' ten ', ' 10 ')]
    for pair in num_pairs:
        desc_lines = [line.replace(pair[0], pair[1]) for line in desc_lines]

    single_quot_regex = re.compile("((?<=\s)'([^']+)'(?=\s))")
    desc_lines = [re.sub(single_quot_regex, r'"\2"', line) for line in desc_lines]

    num_lines = len(regex_lines)
    reps_words = ["dog", "truck", "ring", "lake"]
    reps_tags = ["<M0>", "<M1>", "<M2>", "<M3>"]

    new_regex_lines = ["" for i in range(len(regex_lines))]
    new_desc_lines = ["" for i in range(len(desc_lines))]
    cool = False
    for l_i in range(num_lines):
        desc_line = desc_lines[l_i]
        old_desc = desc_line
        temp_desc_line = ''.join([c for c in desc_line])
        words_replaced = []
        for j in range(4):
            double_quot = re.compile('.*\s"([^"]*)"\s.*')
            double_quot_out = double_quot.match(temp_desc_line)
            if double_quot_out:
                word = double_quot_out.groups()[-1]
                words_replaced.insert(0, word)
                # print(words_replaced)
                temp_desc_line = temp_desc_line.replace('"{}"'.format(word), reps_tags[j])

        for j in range(len(words_replaced)):
            desc_line = desc_line.replace('"{}"'.format(words_replaced[j]), reps_tags[j])

        new_desc_lines[l_i] = desc_line
        regex_line = regex_lines[l_i]
        # print(regex_line)

        # regex_line = regex_line.replace(" ".join('[AEIOUaeiou]'), "<VOW>")
        # regex_line = regex_line.replace(" ".join('[aeiouAEIOU]'), "<VOW>")
        # regex_line = regex_line.replace(" ".join('[0-9]'), "<NUM>")
        # regex_line = regex_line.replace(" ".join('[A-Za-z]'), "<LET>")
        # regex_line = regex_line.replace(" ".join('[A-Z]'), "<CAP>")
        # regex_line = regex_line.replace(" ".join('[a-z]'), "<LOW>")

        regex_line = regex_line.replace(" ".join('AEIOUaeiou'), "<VOW>")
        regex_line = regex_line.replace(" ".join('aeiouAEIOU'), "<VOW>")
        regex_line = regex_line.replace(" ".join('0-9'), "<NUM>")
        regex_line = regex_line.replace(" ".join('A-Za-z'), "<LET>")
        regex_line = regex_line.replace(" ".join('A-Z'), "<CAP>")
        regex_line = regex_line.replace(" ".join('a-z'), "<LOW>")

        for i in range(len(words_replaced)):
            match = re.compile(re.escape(" ".join(words_replaced[i])), re.IGNORECASE)
            # print(match)
            # print(match.sub(" ".join(reps_tags[i]), regex_line))
            regex_line = match.sub(reps_tags[i], regex_line)

        for r_i in range(len(reps_words)):
            r_word = reps_words[r_i]
            regex_line = regex_line.replace(" ".join(r_word), reps_tags[r_i])

        new_regex_lines[l_i] = regex_line


    new_desc_lines = [line.strip(" ") for line in new_desc_lines]
    new_regex_lines = [line.strip(" ") for line in new_regex_lines]
    return new_desc_lines, new_regex_lines

def split_data_and_save(desc_lines, regex_lines, data_dir, with_test):
    regex_lines = [line.rstrip('\n') for line in regex_lines]
    desc_lines = [line.rstrip('\n') for line in desc_lines]

    zipped = list(zip(regex_lines, desc_lines))
    random.shuffle(zipped)
    regex_lines_shuffled, desc_lines_shuffled = zip(*zipped)

    regex_train, regex_val, regex_test = split_train_test_val(regex_lines_shuffled, with_test)
    desc_train, desc_val, desc_test = split_train_test_val(desc_lines_shuffled, with_test)

    with open('{}{}{}.txt'.format(data_dir, "/", "src-train"), "w") as out_file:
        out_file.write("\n".join(desc_train))

    with open('{}{}{}.txt'.format(data_dir, "/", "targ-train"), "w") as out_file:
        out_file.write("\n".join(regex_train))

    with open('{}{}{}.txt'.format(data_dir, "/", "src-val"), "w") as out_file:
        out_file.write("\n".join(desc_val))

    with open('{}{}{}.txt'.format(data_dir, "/", "targ-val"), "w") as out_file:
        out_file.write("\n".join(regex_val))

    if with_test:
        with open('{}{}{}.txt'.format(data_dir, "/", "src-test"), "w") as out_file:
            out_file.write("\n".join(desc_test))

        with open('{}{}{}.txt'.format(data_dir, "/", "targ-test"), "w") as out_file:
            out_file.write("\n".join(regex_test))

    print("Done!")

def split_train_test_val(ar, with_test):
    if with_test:
        ratio = 0.65
        train_set = ar[:int(len(ar)*ratio)]
        not_train_set = ar[int(len(ar)*ratio):]
        val_set = not_train_set[int(len(not_train_set)*(5.0/7.0)):]
        test_set = not_train_set[:int(len(not_train_set)*(5.0/7.0))]

        return train_set, val_set, test_set
    else:
        ratio = 0.75
        train_set = ar[:int(len(ar)*ratio)]
        val_set = ar[int(len(ar)*ratio):]
        return train_set, val_set, []

def output_kushman_format(data_path, output_path):
    desc_lines = [line.rstrip('\n') for line in open('{}/{}'.format(data_path, 'src.txt'))]
    regex_lines = [line.rstrip('\n') for line in open('{}/{}'.format(data_path, 'targ.txt'))]
    # desc_lines = [line.replace('"', '""') for line in desc_lines]

    csv_lines = ['"{}","{}","{}","p","p","p","p","p","n","n","n","n","n"'.format(str(i+1), desc_lines[i], regex_lines[i]) for i in range(len(regex_lines))]
    with open('{}/{}'.format(output_path, "data_kushman_format.csv"), "w") as out_file:
        out_file.write("\n".join(csv_lines))

# dont know why, just copied from original code 
def clean(s):
    PAD = '<blank>'
    BOS = '<s>'
    EOS = '</s>'
    s = s.replace(PAD, '')
    s = s.replace(BOS, '')
    s = s.replace(EOS, '')
    return s

def build_indexer(output_path, unk_threshold=0.0):
    input_word_counts = Counter()
    # Count words and build the indexers
    train_src_lines = read_lines(join(output_path, 'src-train.txt'))
    train_targ_lines = read_lines(join(output_path, 'targ-train.txt'))

    for x in train_src_lines:
        for word in x.split():
            input_word_counts.increment_count(word, 1.0)
    input_indexer = Indexer()
    output_indexer = Indexer()
    # Reserve 0 for the pad symbol for convenience
    input_indexer.get_index(PAD_SYMBOL)
    input_indexer.get_index(UNK_SYMBOL)
    output_indexer.get_index(PAD_SYMBOL)
    output_indexer.get_index(SOS_SYMBOL)
    output_indexer.get_index(EOS_SYMBOL)
    # Index all input words above the UNK threshold
    for word in input_word_counts.keys():
        if input_word_counts.get_count(word) > unk_threshold + 0.5:
            input_indexer.get_index(word)
    # Index all output tokens in train
    for y in train_targ_lines:
        for y_tok in y.split():
            output_indexer.get_index(y_tok)

    input_indexer.save_to_file(join(output_path, 'src-indexer.pkl'))
    output_indexer.save_to_file(join(output_path, 'targ-indexer.pkl'))
    print('Building indexer done')

def do_stats(output_path):
    src_lines = read_lines(join(output_path, 'src.txt'))
    targ_lines = read_lines(join(output_path, 'targ.txt'))
    # unique words
    input_word_counts = Counter()
    for x in src_lines:
        for word in x.split():
            input_word_counts.increment_count(word, 1.0)
    input_indexer = Indexer()
    output_indexer = Indexer()

    for word in input_word_counts.keys():
        input_indexer.get_index(word)
    
    print("Num of Unique words", len(input_indexer))
    print(input_indexer)

    # lens 
    src_lens = [len(x.split()) for x in src_lines]
    src_lens = np.array(src_lens)
    print("Avg, med, input lens", np.mean(src_lens), np.median(src_lens))
    print(np.sum(src_lens > 10) / src_lens.size, np.sum(src_lens > 15) / src_lens.size)
    targ_lens = [len(x.split()) for x in targ_lines]
    targ_lens = np.array(targ_lens)
    print("Avg, med, max, output lens", np.mean(targ_lens), np.median(targ_lens), np.max(targ_lens))
    print(np.sum(targ_lens > 20) / src_lens.size, np.sum(targ_lens > 25) / src_lens.size, np.sum(targ_lens > 30) / src_lens.size)

    print("Top 5")
    n = 5
    # idx = np.arange(len(src_lines))
    idx = np.argsort(src_lens)
    print(src_lens[idx[-n:]])
    for i in range(1, n + 1):
        print(src_lines[idx[-i]], targ_lines[idx[-i]])

    idx = np.argsort(targ_lens)
    print(targ_lens[idx[-n:]])

    for i in range(1, n + 1):
        print(src_lines[idx[-i]], targ_lines[idx[-i]])

    # targ_lens 
    top_percent = 25
    print("Top %d lens----------------------------" % (top_percent))
    topqutile_threshold = np.quantile(targ_lens, 1 - top_percent * 0.01)
    print(topqutile_threshold)
    top_targ_lens = targ_lens[targ_lens > topqutile_threshold]
    print(np.mean(top_targ_lens), top_targ_lens.size)


    # measure depth
    print("Top depth -------------")


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data
    if args.do_index:
        output_path = join('./datasets', args.dataset)
        build_indexer(output_path)
        exit()
    if args.do_stats:
        output_path = join('./datasets', args.dataset)
        do_stats(output_path)
        exit()

    data_path = join('./data', args.dataset)
    src_lines, targ_lines = process_tokens(data_path)

    # don't know why, just show some respect to origin code
    src_lines = [clean(x) for x in src_lines]
    targ_lines = [clean(x) for x in targ_lines]

    output_path = join('./datasets', args.dataset)
    with open(join(output_path, "src.txt"), "w") as f:
        for i in src_lines:
            f.write(i + "\n")
    with open(join(output_path, "targ.txt"), "w") as f:
        for i in targ_lines:
            f.write(i + "\n")
    exit()
    split_data_and_save(src_lines, targ_lines, output_path, args.with_test)
    output_kushman_format(data_path, output_path)

    # tokenize build indexer
    build_indexer(output_path)

