import argparse
import random
import numpy as np
import time
import torch
from torch import optim
# from lf_evaluator import *
from models import *
from data import *
from utils import *
import math
from external.regexDFAEquals import dfa_eual_test
from os.path import join
from gadget import *
import os
import shutil

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('dataset', help='specified dataset')
    parser.add_argument('model_id', help='specified model id')
    
    parser.add_argument('--split', type=str, default='test', help='test split')
    parser.add_argument('--do_testbeam', dest='do_testbeam', default=False, action='store_true', help='test entire beam')
    parser.add_argument('--do_output', dest='do_output', default=False, action='store_true', help='only output')
    parser.add_argument('--outfile', dest='outfile', default='beam_output.txt', help='output file of beam')
    # parser.add_argument('--outfolder', dest='outfolder', default='./beam_output', help='output folder')

    # Some common arguments for your convenience
    parser.add_argument('--gpu', type=str, default=None, help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--beam_size', type=int, default=20, help='beam size')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--reverse_input', dest='reverse_input', default=False, action='store_true')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')
    args = parser.parse_args()
    return args

def make_input_tensor(exs, reverse_input):
    x = np.array(exs.x_indexed)
    len_x = len(exs.x_indexed)
    if reverse_input:
        x = np.array(x[::-1])
    # add batch dim
    x = x[np.newaxis, :]
    len_x = np.array([len_x])
    x = torch.from_numpy(x).long()
    len_x = torch.from_numpy(len_x)
    return x, len_x

def test_model(model_path, test_data, input_indexer, output_indexer, args):
    device = config.device
    if 'cpu' in str(device):
        checkpoint = torch.load(model_path, map_location=device)
    else:
        checkpoint = torch.load(model_path)
    
    #  Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = AttnRNNDecoder(args.input_dim, args.hidden_size, 2 * args.hidden_size if args.bidirectional else args.hidden_size,len(output_indexer), args.rnn_dropout)

    # load dict
    model_input_emb.load_state_dict(checkpoint['input_emb'])
    model_enc.load_state_dict(checkpoint['enc'])
    model_output_emb.load_state_dict(checkpoint['output_emb'])
    model_dec.load_state_dict(checkpoint['dec'])

    # map device
    model_input_emb.to(device)
    model_enc.to(device)
    model_output_emb.to(device)
    model_dec.to(device)

    # switch to eval
    model_input_emb.eval()
    model_enc.eval()
    model_output_emb.eval()
    model_dec.eval()

    pred_derivations = []
    with torch.no_grad():
        for i, ex in enumerate(test_data):
            if i % 50 == 0:
                print("Done", i)
            x, len_x = make_input_tensor(ex, args.reverse_input)
            x, len_x = x.to(device), len_x.to(device)

            enc_out_each_word, enc_context_mask, enc_final_states = \
                    encode_input_for_decoder(x, len_x, model_input_emb, model_enc)
            
            pred_derivations.append(beam_decoder(enc_out_each_word, enc_context_mask, enc_final_states,
                output_indexer, model_output_emb, model_dec, args.decoder_len_limit, args.beam_size))

    # if args.do_output:
    #     if args.do_testbeam:
    #         output_derivations(test_data, pred_derivations, args, out_to_folder=True)
    #     else:
    #         output_derivations(test_data, pred_derivations, args)
    # else:
    #     if args.do_testbeam:
    #         evaluate_beam(test_data, pred_derivations, args)
    #     else:
    #         evaluate(test_data, pred_derivations, outfile=args.outfile)

    output_derivations(test_data, pred_derivations, args, out_to_folder=True)

def beam_decoder(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                    model_output_emb, model_dec, decoder_len_limit, beam_size):
    device = config.device
    EOS = output_indexer.get_index(EOS_SYMBOL)
    context_inf_mask = get_inf_mask(enc_context_mask)
    dec_hidden_states = enc_final_states
    
    current_beam = Beam(1)
    current_beam.add((dec_hidden_states, output_indexer.index_of(SOS_SYMBOL), []), 0)

    for _ in range(decoder_len_limit):
        next_beam = Beam(beam_size)
        for ((state, tok_to_feed, y_toks), score) in current_beam.get_elts_and_scores():
            if tok_to_feed == EOS:
                # do nothing
                next_beam.add((state, tok_to_feed, y_toks), score)
                continue

            # tok not EOS
            input_words = torch.from_numpy(np.asarray([[tok_to_feed]])).to(device)
            input_embeded_words = model_output_emb.forward(input_words)
            voc_scores, next_state = model_dec(input_embeded_words, state, enc_out_each_word, context_inf_mask)
            voc_scores = voc_scores.cpu().numpy().flatten()
            voc_scores = np.log(voc_scores)
            for (voc_id, voc_score) in enumerate(voc_scores):
                next_beam.add((next_state, voc_id, y_toks+[voc_id]), score + voc_score)
        current_beam = next_beam
    
    # transform to deriviations
    pred_tokens = []
    for ((_, _, y_toks), score) in current_beam.get_elts_and_scores():
        tokens = []
        for word in y_toks:
            if word == EOS:
                break
            tok = output_indexer.get_object(word)
            tokens.append(tok)
        pred_tokens.append(tokens)
    return pred_tokens

def makedir_f(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

def output_derivations(test_data, pred_derivations, args, out_to_folder=False):

    if out_to_folder:
        outfolder = join('decodes/', args.dataset, '{}-{}'.format(args.split, args.model_id))
        makedir_f(outfolder)
        for i, pred_ders in enumerate(pred_derivations):
            lines = "\n".join(["{} {}".format(x[0], "".join(x[1])) for x in enumerate(pred_ders)])
            with open(join(outfolder, str(i + 1)), "w") as out:
                out.writelines(lines)
    else:
        selected_derivs = [x[0] for x in pred_derivations]
        # Writes to the output file if needed
        with open(args.outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write("".join(selected_derivs[i]) + "\n")

def evaluate(test_data, pred_derivations, print_output=True, outfile=None, show_example=False):
    # e = GeoqueryDomain()
    # print(pred_derivations)
    # selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    selected_derivs = [x[0] for x in pred_derivations]
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    pred_match = []
    for i, ex in enumerate(test_data):
        print("----------------------------", ex.id, "-------------------------")
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i])
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i], ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if  dfa_eual_test(' '.join(ex.y_tok), ' '.join(selected_derivs[i])):
            num_denotation_match += 1
            pred_match.append(1)
        else:
            pred_match.append(0)

    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(ex.y_tok) + "\n")
                out.write(" ".join(selected_derivs[i]) + "\t" + str(pred_match[i]) + "\n")
        out.close()

def evaluate_beam(test_data, pred_derivations, args):
    pred_tokens = []
    pred_extact_match = []
    pred_deno_match = []
    pred_match = []

    beam_size = args.beam_size
    for i, ex in enumerate(test_data):
        i_pred_tokens = []
        i_pred_match = []
        i_exact_match = beam_size
        i_deno_match = beam_size
        for j in range(args.beam_size):
            # Compute accuracy metrics
            y_pred = ' '.join(pred_derivations[i][j])
            i_pred_tokens.append(y_pred)
            # Check exact match
            if y_pred == ' '.join(ex.y_tok):
                i_exact_match = min(j, i_exact_match)
            # Check position-by-position token correctness
            if  dfa_eual_test(' '.join(ex.y_tok), y_pred):
                i_deno_match = min(j, i_deno_match)
                i_pred_match.append(1)
            else:
                i_pred_match.append(0)


        pred_tokens.append(i_pred_tokens)
        pred_extact_match.append(i_exact_match)
        pred_deno_match.append(i_deno_match)
        pred_match.append(i_pred_match)

    cut_offs = [1,3,5,10,25]
    print("Num Data ", len(test_data))
    for cut_off in cut_offs:
        num_exact_match = np.sum(np.asarray([i < cut_off for i in pred_extact_match]))
        num_deno_match = np.sum(np.asarray([i < cut_off for i in pred_deno_match]))
        print(cut_off, num_exact_match, num_deno_match)
    # Writes to the output file if needed
    outfile = args.outfile
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(ex.y_tok) + "\n")
                for j in range(args.beam_size):
                    out.write(str(j) + "\t" + pred_tokens[i][j] + "\t" + str(pred_match[i][j]) +"\n")
        out.close()

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    # global device
    set_global_device(args.gpu)
    
    print("Pytroch using device ", config.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data
    test, input_indexer, output_indexer = load_test_dataset(args.dataset, args.split)
    test_data_indexed = index_data(test, input_indexer, output_indexer, args.decoder_len_limit)
    # test_data_indexed = tricky_filter_data(test_data_indexed)
    print(len(test_data_indexed))
    print("%i test exs, %i input types, %i output types" % (len(test_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)

    print("=======FINAL EVALUATION ON BLIND TEST=======")
    model_path = get_model_file(args.dataset, args.model_id)
    test_model(model_path, test_data_indexed, input_indexer, output_indexer, args)