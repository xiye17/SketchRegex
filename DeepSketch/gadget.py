
import torch
import sys
from torch import optim
# from lf_evaluator import *
from models import *
from data import *
from utils import *
import multiprocessing as mp
from SynthCache import SynthWorker

class config():
    device = None

def set_global_device(gpu):
    if gpu is not None:
        config.device = torch.device(('cuda:' + gpu) if torch.cuda.is_available() else 'cpu')
    else:
        config.device = 'cpu'
    
# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

def masked_cross_entropy(voc_scores, gt, mask):
    corss_entropy = -torch.log(torch.gather(voc_scores, 1, gt.view(-1, 1)))
    loss = corss_entropy.squeeze(1).masked_select(mask.byte()).sum()
    return loss

# Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
# inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    # print(enc_output_each_word.size(), enc_context_mask.size())
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)

def orcale_reward(batch_tokens, batch_ids, split, cache, output_indexer):
    batch_ids = batch_ids.numpy()

    batch_results = []
    for i, tokens in enumerate(batch_tokens):
        single_results = []

        for j, seq in enumerate(tokens): 
            sketch = "".join([output_indexer.get_object(x) for x in seq])
            result = cache.query(split, batch_ids[i], sketch)
            print(split, batch_ids[i], sketch, result, file=sys.stderr)
            single_results.append(result)
        batch_results.append(single_results)
    
    # batch_rewards 1 - 0 set
    # batch_coverage
    # batch_match
    num_match = 0
    for single_results in batch_results:
        for res in single_results:
            if res == "false":
                break
            if res == "true":
                num_match += 1
                break

    batch_rewards = np.array([[1 if j == "true" else 0 for j in i] for i in batch_results])
    num_coverage = int(np.sum(np.sum(batch_rewards, axis=1) > 0))

    return batch_rewards, num_coverage, num_match

def dfa_orcale_reward(batch_tokens, batch_ids, split, cache, output_indexer):
    batch_results = []
    EOS = output_indexer.get_index(EOS_SYMBOL)
    batch_gts = batch_ids.numpy()
    for i, tokens in enumerate(batch_tokens):
        single_results = []
        gold = []
        for x in batch_gts[i]:
            if x == EOS:
                break
            gold.append(output_indexer.get_object(x))
        gold = "".join(gold)
        for j, seq in enumerate(tokens): 
            pred = "".join([output_indexer.get_object(x) for x in seq])
            result = cache.query(gold, pred)
            single_results.append(result)
            # print(i, j, pred, result)
        batch_results.append(single_results)
    
    num_match = 0
    for single_results in batch_results:
        res = single_results[0]
        if res != "false":
            num_match += 1

    batch_rewards = np.array([[1 if j != "false" else 0 for j in i] for i in batch_results])
    num_coverage = int(np.sum(np.sum(batch_rewards, axis=1) > 0))
    return batch_rewards, num_coverage, num_match

def parallel_orcale_reward(batch_tokens, batch_ids, split, cache, output_indexer):
    batch_ids = batch_ids.numpy()

    batch_results = []

    id_pool = []
    to_test_pool = []
    for i, tokens in enumerate(batch_tokens):
        single_results = []

        for j, seq in enumerate(tokens): 
            sketch = "".join([output_indexer.get_object(x) for x in seq])
            result = cache.soft_query(split, batch_ids[i], sketch)
            if result is None:
                id_pool.append((i, j))
                to_test_pool.append((batch_ids[i], sketch))

            single_results.append(result)
        batch_results.append(single_results)
    
    print("Pool Size", len(to_test_pool), file=sys.stderr)
    dataset = cache.dataset
    worker = SynthWorker(dataset, split)
    pool = mp.Pool(5)
    results_pool = pool.map(worker.run, to_test_pool)
    pool.close()

    for res_id, to_test in enumerate(to_test_pool):
        batch_results[id_pool[res_id][0]][id_pool[res_id][1]] = results_pool[res_id]
        cache.soft_write(split, to_test[0], to_test[1], results_pool[res_id])
        # print(to_test[0], to_test[1], results_pool[res_id], file=sys.stderr)

    # batch_rewards 1 - 0 set
    # batch_coverage
    # batch_match
    num_match = 0
    for single_results in batch_results:
        for res in single_results:
            if res == "false":
                break
            if res == "true":
                num_match += 1
                break

    batch_rewards = np.array([[1 if j == "true" else 0 for j in i] for i in batch_results])
    num_coverage = int(np.sum(np.sum(batch_rewards, axis=1) > 0))

    return batch_rewards, num_coverage, num_match

def parallel_synth(test_data, pred_derivations, split, cache):
    batch_results = []

    id_pool = []
    to_test_pool = []
    for i, tokens in enumerate(pred_derivations):
        single_results = []

        for j, seq in enumerate(tokens): 
            sketch = seq
            # result = cache.query(split, test_data[i].id, sketch)
            result = cache.soft_query(split, test_data[i].id, sketch)
            if result is None:
                id_pool.append((i, j))
                to_test_pool.append((test_data[i].id, sketch))
            else:
                # print(split, test_data[i].id, sketch, result, file=sys.stderr)
                pass

            single_results.append(result)
        batch_results.append(single_results)
    
    print("Pool Size", len(to_test_pool))
    timeout = cache.timeout
    worker = SynthWorker(timeout, split)
    pool = mp.Pool(5)
    results_pool = pool.map(worker.run, to_test_pool)
    pool.close()

    for res_id, to_test in enumerate(to_test_pool):
        batch_results[id_pool[res_id][0]][id_pool[res_id][1]] = results_pool[res_id]
        cache.soft_write(split, to_test[0], to_test[1], results_pool[res_id])
        # print(split, to_test[0], to_test[1], batch_results[id_pool[res_id][0]][id_pool[res_id][1]], file=sys.stderr)

    return batch_results
