# make distant supervision dataset
from SynthCache import *
from data import *
from os.path import join
from eval import read_derivations
import multiprocessing as mp

# def _parse_args():
#     parser = argparse.ArgumentParser(description='main.py')

#     parser.add_argument('dataset', help='specified dataset')
#     parser.add_argument('model_id', help='specified model id')
#     parser.add_argument('--split', type=str, default='test', help='test split')
#     parser.add_argument('--cache_id', type=str, default="cache", help='cache_id')
#     parser.add_argument('--decoder_len_limit', type=int, default=50, help='output length limit of the decoder')

#     args = parser.parse_args()
#     return args


# def parallel_oracle_evaluate(test_data, pred_derivations, split, cache):
#     batch_results = []

#     id_pool = []
#     to_test_pool = []
#     for i, tokens in enumerate(pred_derivations):
#         single_results = []

#         for j, seq in enumerate(tokens): 
#             sketch = "".join(seq)
#             # result = cache.query(split, test_data[i].id, sketch)
#             result = cache.soft_query(split, test_data[i].id, sketch)
#             if result is None:
#                 id_pool.append((i, j))
#                 to_test_pool.append((test_data[i].id, sketch))

#             single_results.append(result)
#         batch_results.append(single_results)
    
#     print("Pool Size", len(to_test_pool))
#     dataset = cache.dataset
#     worker = SynthWorker(dataset, split)
#     pool = mp.Pool(5)
#     results_pool = pool.map(worker.run, to_test_pool)
#     pool.close()

#     for res_id, to_test in enumerate(to_test_pool):
#         batch_results[id_pool[res_id][0]][id_pool[res_id][1]] = results_pool[res_id]
#         cache.soft_write(split, to_test[0], to_test[1], results_pool[res_id])
#     print_stats(batch_results)

def precache_for_one_example(ex, preds, split, timed_cache):

    id_pool = []
    to_test_pool = []
    single_results = []
    for j, seq in enumerate(preds): 
        sketch = "".join(seq)
        # result = cache.query(split, test_data[i].id, sketch)
        result = timed_cache.timed_soft_query(split, ex.id, sketch)
        if result is None:
            id_pool.append(j)
            to_test_pool.append((ex.id, sketch))

        single_results.append(result)

    # print("Pool Size", len(to_test_pool))
    dataset = timed_cache.dataset
    worker = SynthWorker(dataset, split)
    pool = mp.Pool(5)
    results_pool = pool.map(worker.timed_run, to_test_pool)
    pool.close()

    for res_id, to_test in enumerate(to_test_pool):
        single_results[id_pool[res_id]] = results_pool[res_id]
        timed_cache.soft_write(split, to_test[0], to_test[1], results_pool[res_id])

    print(single_results)

def check_empty(split, ex):
    fname = join('external', 'examples', 'turk', 'example-{}'.format(split), str(ex.id))
    with open(fname) as f:
        lines = f.readlines()
    if 'null' in lines[1]:
        return True
    else:
        return False

class Args:
    pass

def precache_train_synth():
    args = Args()
    args.dataset = 'TurkSketch'
    args.split = 'train'
    args.oracle_mode = 'sketch'
    args.cache_id = 'cache'
    args.decoder_len_limit = 50

    cutoff = 20

    cache = TimedCache(args.cache_id, args.dataset)

    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)
    test_data_indexed = index_data(test, input_indexer, output_indexer, args.decoder_len_limit)
    # test_data_indexed = filter_data(test_data_indexed)

    decode_folder = join('external/', 'train-gramgen')
    pred_derivations = read_derivations(decode_folder, test_data_indexed)

    # parallel_oracle_evaluate(test_data_indexed, pred_derivations, args.split, cache)
    
    assert len(test_data_indexed) == len(pred_derivations)
    for i, (ex, preds) in enumerate(zip(test_data_indexed, pred_derivations)):
        # is empty
        if check_empty(args.split, ex):
            continue

        precache_for_one_example(ex, preds[:cutoff], args.split, cache)
        if (i + 1) % 50 == 0:
            print(i + 1)
            cache.rewrite()

    # for k in cache.data:
    #     print(k)
    #     print(cache.data[k])

    cache.rewrite()

def tokenize_spec(x):
    y = []
    while len(x) > 0:
        head = x[0]
        if head in ["?", "(", ")", "{", "}", ","]:
            y.append(head)
            x = x[1:]
        elif head == "<":
            end = x.index(">") + 1
            y.append(x[:end])
            x = x[end:]
        else:
            leftover = [(i in ["?", "(", ")", "{", "}", "<", ">"]) for i in x]
            end = leftover.index(True)
            y.append(x[:end])
            x = x[end:]
    return " ".join(y)

def gt_for_one_example(ex, preds, split, timed_cache):
    if ex.y == 'null':
        return 'null'
    if check_empty(split, ex):
        return 'empty'

    single_results = []
    for j, seq in enumerate(preds): 
        sketch = "".join(seq)
        # result = cache.query(split, test_data[i].id, sketch)
        result = timed_cache.timed_soft_query(split, ex.id, sketch)
        single_results.append(result)
    
    assert (all([x is not None for x in single_results]))
    pairs = zip(preds, single_results)
    # print("Pool Size", len(to_test_pool))
    pairs = [(x, y) for (x, y) in pairs if y[0] == 'true']

    if not pairs:
        return 'none'
    p_long = pairs[0]
    # sort_by_time
    pairs.sort(key=lambda x: x[1][1])
    p_quick = pairs[0]

    p = p_long
    return tokenize_spec(p[0])
    
def compose_train_synth():
    args = Args()
    args.dataset = 'TurkSketch'
    args.split = 'train'
    args.oracle_mode = 'sketch'
    args.cache_id = 'cache'
    args.decoder_len_limit = 50

    cutoff = 50

    cache = TimedCache(args.cache_id, args.dataset)

    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)
    test_data_indexed = index_data(test, input_indexer, output_indexer, args.decoder_len_limit)
    # test_data_indexed = filter_data(test_data_indexed)

    decode_folder = join('external/', 'train-gramgen')
    pred_derivations = read_derivations(decode_folder, test_data_indexed)

    # parallel_oracle_evaluate(test_data_indexed, pred_derivations, args.split, cache)
    
    assert len(test_data_indexed) == len(pred_derivations)
    
    new_specs = []
    for i, (ex, preds) in enumerate(zip(test_data_indexed, pred_derivations)):
        # is empty
        gt = gt_for_one_example(ex, preds[:cutoff], args.split, cache)
        new_specs.append(gt)
    # for k in cache.data:
    #     print(k)
    #     print(cache.data[k])

    with open('targ-train.txt', 'w') as f:
        f.writelines([x + '\n' for x in new_specs])

if __name__ == "__main__":
    # precache_train_synth()
    compose_train_synth()
    
