from SynthCache import *
from data import *
from os.path import join
import numpy as np
import multiprocessing as mp
import argparse
from external.regexDFAEquals import dfa_eual_test
import sys
import subprocess


def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('dataset', help='specified dataset')
    parser.add_argument('model_id', help='specified model id')
    parser.add_argument('--split', type=str, default='test', help='test split')
    parser.add_argument('--cache_id', type=str, default="cache", help='cache_id')
    parser.add_argument('--decoder_len_limit', type=int, default=50, help='output length limit of the decoder')
    parser.add_argument('--do_filter', default=False, action='store_true', help='run filtering on regex dataset')

    args = parser.parse_args()
    return args

def print_stats(stats):
    first_true = []
    first_false = []
    cover = []
    empty = []
    null = []
    
    for (i, results) in enumerate(stats):
        if not results:
            empty.append(i)
        for res in results:
            if res == "false":
                first_false.append(i)
                break
            if res == "true":
                first_true.append(i)
                break
            if res == "empty":
                empty.append(i)
                break
        if any([x == "true" for x in results]):
            cover.append(i)

        if all([(x == "null" or x == "wrong" or x == "timeout") for x in results]):
            null.append(i)
    
    # easy_print = lambda x, y: print(x, len(y))
    # easy_print("First true", first_true)
    # easy_print("First false", first_false)
    # easy_print("Cover", cover)
    # easy_print("Empty", empty)
    # easy_print("Null", null)
    print('wrong synthesized results: {:.3f}'.format(len(first_false)/len(stats)))
    print('time-out: {:.3f}'.format(len(null)/len(stats)))
    print('semantic acc: {:.3f}'.format((len(first_true) + len(empty))/len(stats)))

def debug_stats(stats):
    first_true = []
    first_false = []
    cover = []
    empty = []
    null = []
    
    for (i, results) in enumerate(stats):
        if not results:
            empty.append(i)
        for res in results:
            if res == "false":
                print(i + 1, 'False', results)
                first_false.append(i)
                break
            if res == "true":
                first_true.append(i)
                break
            if res == "empty":
                empty.append(i)
                break
        if any([x == "true" for x in results]):
            cover.append(i)

        if all([(x == "null" or x == "wrong" or x == "timeout") for x in results]):
            print(i + 1, 'Null', results)
            null.append(i)
    
    easy_print = lambda x, y: print(x, len(y))
    easy_print("First true", first_true)
    easy_print("First false", first_false)
    easy_print("Cover", cover)
    easy_print("Empty", empty)
    easy_print("Null", null)
    print('{:.3f}'.format((len(first_true) + len(empty))/len(stats)))
    
    # print
    # first false
    print('First False', [a + 1 for a in first_false])
    print('Null', [a + 1 for a in null])

    print('\n\n')
    for (i, results) in enumerate(stats):
        if not results:
            empty.append(i)
        print(i + 1, results)


def parallel_oracle_evaluate(test_data, pred_derivations, split, cache):
    batch_results = []

    id_pool = []
    to_test_pool = []
    for i, tokens in enumerate(pred_derivations):
        single_results = []

        for j, seq in enumerate(tokens): 
            sketch = "".join(seq)
            # result = cache.query(split, test_data[i].id, sketch)
            result = cache.soft_query(split, test_data[i].id, sketch)
            if result is None:
                id_pool.append((i, j))
                to_test_pool.append((test_data[i].id, sketch))

            single_results.append(result)
        batch_results.append(single_results)
    
    print("Pool Size", len(to_test_pool))
    dataset = cache.dataset
    worker = SynthWorker(dataset, split)
    pool = mp.Pool(5)
    results_pool = pool.map(worker.run, to_test_pool)
    pool.close()

    for res_id, to_test in enumerate(to_test_pool):
        batch_results[id_pool[res_id][0]][id_pool[res_id][1]] = results_pool[res_id]
        cache.soft_write(split, to_test[0], to_test[1], results_pool[res_id])
    print_stats(batch_results)

def read_sketches(filename):
    if not os.path.exists(filename):
        return [["0","?"]] * 20
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip().split() for x in lines]
        
        lines = [(x[0], "") if len(x) == 1 else x for x in lines]
    return lines

def read_derivations(prefix, data):
    ders = []
    for i, d in enumerate(data):
        sketches = read_sketches(join(prefix, str(d.id)))
        if sketches:
            ders.append([x[1] for x in sketches])
        else:
            ders = []
            
    return ders

def dfa_acc_evaluate(test_data, pred_derivations, split, cache):
    selected_derivs = [x[0] for x in pred_derivations]
    num_exact_match = 0
    num_denotation_match = 0

    for i, ex in enumerate(test_data):
        y_pred = ''.join(selected_derivs[i])
        gold  = ''.join(ex.y_tok)
        if y_pred == gold:
            num_exact_match += 1

        # Check correctness of the denotation
        result = cache.query(gold, y_pred)
        if  result != 'false':
            num_denotation_match += 1

    print("exact-match acc: %s" % (render_ratio(num_exact_match, len(test_data))))
    print("semantic acc: %s" % (render_ratio(num_denotation_match, len(test_data))))
    
def run_filtering_test(args):
    print('Run filter')
    if args.dataset == 'KB13':
        mode = 'kb13'
        dataset_id = 'kb13'
    elif args.dataset == 'Turk':
        mode = 'dr'
        dataset_id = 'turk'
    else:
        raise RuntimeError('Dataset is not supposed to run filtering test')
    
    example_path = join('external/examples/', dataset_id, f'example-{args.split}') + '/'
    decodes_path = join('decodes/', args.dataset, '{}-{}'.format(args.split, args.model_id)) + '/'
    cmd = ["java", "-cp", "external/run_filter.jar:external/lib/*", "-ea", "datagen.Main", mode, 'filter', example_path , decodes_path]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    out = out.decode('utf-8')


    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)
    test_data_indexed = index_data(test, input_indexer, output_indexer, args.decoder_len_limit)
    test_data_indexed = filter_data(test_data_indexed)

    # invalid ones
    out = out.split()
    # print(out)
    coverage = int(out[0])
    match = int(out[1])

    # print(coverage, out)
    print('consistent found: {:.3f}'.format(coverage/len(test)))
    print('semantic ac: {:.3f}'.format(match/len(test)))

if __name__ == "__main__":

    args = _parse_args()
    print(args)
    args.oracle_mode = 'sketch' if 'Sketch' in args.dataset else 'regex'

    if args.oracle_mode == 'sketch':
        cache = SynthCache(args.cache_id, args.dataset)
    else:
        if args.do_filter:
            run_filtering_test(args)
            exit()
        cache = DFACache(args.cache_id, args.dataset)

    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)
    test_data_indexed = index_data(test, input_indexer, output_indexer, args.decoder_len_limit)
    test_data_indexed = filter_data(test_data_indexed)

    decode_folder = join('decodes/', args.dataset, '{}-{}'.format(args.split, args.model_id))
    pred_derivations = read_derivations(decode_folder, test_data_indexed)

    if args.oracle_mode == 'sketch':
        parallel_oracle_evaluate(test_data_indexed, pred_derivations, args.split, cache)
    else:
        dfa_acc_evaluate(test_data_indexed, pred_derivations, args.split, cache)
    cache.rewrite()
