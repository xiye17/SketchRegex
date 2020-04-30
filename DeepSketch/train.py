import argparse
import random
import sys
import numpy as np
import time
import torch
from torch import optim
from gadget import *
from models import *
from data import *
from utils import *
from SynthCache import *
import math
from external.regexDFAEquals import dfa_eual_test

cache = None

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('dataset', help='specified dataset')
    # General system running and configuration options

    # Some common arguments for your convenience
    parser.add_argument("--gpu", type=str, default="0", help="gpu id")
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--clip_grad', type=float, default=10.0)

    # regarding model saving
    parser.add_argument('--model_id', type=str, default=None, help='model identifier')
    parser.add_argument('--saving_from', type=int, default=50, help='saving from - epoch')
    parser.add_argument('--saving_interval', type=int, default=10, help='saving iterval')

    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=50, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--reverse_input', dest='reverse_input', default=False, action='store_true')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')

    # control reinforcement learning
    parser.add_argument('--do_rl', dest='do_rl', default=False, action='store_true', help='do reinforcement learning')
    parser.add_argument('--oracle_mode', type=str, default="sketch", help='oracle')
    parser.add_argument('--do_montecarlo', default=False, action='store_true', help='do montecarlo sampling')
    parser.add_argument('--sample_size', type=int, default=10, help='num sample')
    parser.add_argument('--start_size', type=int, default=0, help='num sample of warm start')
    parser.add_argument('--epoch_start', type=int, default=20, help='epoch of starting')
    parser.add_argument('--warm_model_id', type=str, default=None, help='warm start model')
    parser.add_argument('--cache_id', type=str, default="cache", help='cache_id')
    parser.add_argument('--timeout', type=int, default=2, help='timeout')

    args = parser.parse_args()
    return args


def train_decode_with_output_of_encoder(enc_out_each_word, enc_context_mask,
                            enc_final_states, output_indexer, gt_out, gt_out_lens,
                            model_output_emb, model_dec, decoder_len_limit, p_forcing):
    batch_size = enc_context_mask.size(0)
    context_inf_mask = get_inf_mask(enc_context_mask)
    input_words = torch.from_numpy(np.asarray([output_indexer.index_of(SOS_SYMBOL) for _ in range(batch_size)]))
    input_words = input_words.to(config.device)
    input_words = input_words.unsqueeze(1)
    dec_hidden_states = enc_final_states

    gt_out_mask = sent_lens_to_mask(gt_out_lens, gt_out.size(1))
    output_max_len = torch.max(gt_out_lens).item()

    using_teacher_forcing = np.random.uniform() < p_forcing
    loss = 0

    if using_teacher_forcing:
        for i in range(output_max_len):
            input_embeded_words = model_output_emb.forward(input_words)
            input_embeded_words = input_embeded_words.reshape((1, batch_size, -1))
            voc_scores, dec_hidden_states = model_dec(input_embeded_words, dec_hidden_states, enc_out_each_word, context_inf_mask)
            input_words = gt_out[:, i].view((-1, 1))

            loss += masked_cross_entropy(voc_scores, gt_out[:, i], gt_out_mask[:, i])

    else:
        for i in range(output_max_len):
            input_embeded_words = model_output_emb.forward(input_words)
            input_embeded_words = input_embeded_words.reshape((1, batch_size, -1))
            voc_scores, dec_hidden_states = model_dec(input_embeded_words, dec_hidden_states, enc_out_each_word, context_inf_mask)
            output_words = voc_scores.argmax(dim=1, keepdim=True)
            input_words = output_words.detach()
            loss += masked_cross_entropy(voc_scores, gt_out[:, i], gt_out_mask[:, i])

    num_entry = gt_out_lens.sum().float().item()
    loss = loss / num_entry
    return loss, num_entry

def model_perplexity(test_loader,
                    model_input_emb, model_enc, model_output_emb, model_dec,
                    input_indexer, output_indexer, args):
    device = config.device
    model_input_emb.eval()
    model_enc.eval()
    model_output_emb.eval()
    model_dec.eval()

    test_iter = iter(test_loader)
    epoch_loss = 0.0
    epoch_num_entry = 0.0

    with torch.no_grad():
        for _, batch_data in enumerate(test_iter):
            batch_in, batch_in_lens, batch_out, batch_out_lens = batch_data
            batch_in, batch_in_lens, batch_out, batch_out_lens = \
                batch_in.to(device), batch_in_lens.to(device), batch_out.to(device), batch_out_lens.to(device)

            enc_out_each_word, enc_context_mask, enc_final_states = \
                encode_input_for_decoder(batch_in, batch_in_lens, model_input_emb, model_enc)

            loss, num_entry = \
                train_decode_with_output_of_encoder(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                batch_out, batch_out_lens, model_output_emb, model_dec, args.decoder_len_limit, 1)
            epoch_loss += (loss.item() * num_entry)
            epoch_num_entry += num_entry
        perperlexity = epoch_loss / epoch_num_entry
    return perperlexity

def train_model_encdec_ml(train_data, test_data, input_indexer, output_indexer, args):
    device = config.device
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    train_input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    test_input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
    input_max_len = max(train_input_max_len, test_input_max_len)

    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    train_output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    test_output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in test_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, args.decoder_len_limit)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer,  np.max(np.asarray([len(ex.y_indexed) for ex in test_data])) )
    all_test_output_data = np.maximum(all_test_output_data, 0)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = AttnRNNDecoder(args.input_dim, args.hidden_size, 2 * args.hidden_size if args.bidirectional else args.hidden_size,len(output_indexer), args.rnn_dropout)

    model_input_emb.to(device)
    model_enc.to(device)
    model_output_emb.to(device)
    model_dec.to(device)

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters

    # optimizer = None
    train_loader = BatchDataLoader(train_data, all_train_input_data, all_train_output_data, batch_size=args.batch_size, shuffle=True)
    test_loader = BatchDataLoader(test_data, all_test_input_data, all_test_output_data, batch_size=args.batch_size, shuffle=False)

    train_iter = iter(train_loader)

    optimizer = optim.Adam([
        {'params': model_input_emb.parameters()},
        {'params': model_enc.parameters()},
        {'params': model_output_emb.parameters()},
        {'params': model_dec.parameters()}], lr=0.001)

    get_teaching_forcing_ratio = lambda x: 1.0
    clip = args.clip_grad

    best_dev_perplexity = np.inf
    for epoch in range(1, args.epochs + 1):

        model_input_emb.train()
        model_enc.train()
        model_output_emb.train()
        model_dec.train()

        print('epoch {}'.format(epoch))
        epoch_loss = 0.0
        epoch_num_entry = 0.0
        for batch_idx, batch_data in enumerate(train_iter):

            optimizer.zero_grad()

            batch_in, batch_in_lens, batch_out, batch_out_lens = batch_data
            batch_in, batch_in_lens, batch_out, batch_out_lens = \
                batch_in.to(device), batch_in_lens.to(device), batch_out.to(device), batch_out_lens.to(device)

            enc_out_each_word, enc_context_mask, enc_final_states = \
                encode_input_for_decoder(batch_in, batch_in_lens, model_input_emb, model_enc)

            tf_ratio = get_teaching_forcing_ratio(epoch)
            loss, num_entry = \
                train_decode_with_output_of_encoder(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                batch_out, batch_out_lens, model_output_emb, model_dec, args.decoder_len_limit, tf_ratio)

            loss.backward()
            epoch_loss += (loss.item() * num_entry)
            epoch_num_entry += num_entry
            # print('epoch loss', epoch_loss, 'epoch entry', epoch_num_entry)
            _ = torch.nn.utils.clip_grad_norm_(model_input_emb.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_enc.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_output_emb.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_dec.parameters(), clip)
            optimizer.step()

        print('epoch {} tf: {} train loss: {}'.format(epoch, tf_ratio, epoch_loss / epoch_num_entry))

        if (epoch < args.saving_from) or (args.model_id is None):
            continue

        # start saving
        dev_perplexity = model_perplexity(test_loader, model_input_emb, model_enc, model_output_emb, model_dec, input_indexer, output_indexer, args)
        print('epoch {} tf: {} dev loss: {}'.format(epoch, tf_ratio, dev_perplexity))

        if dev_perplexity < best_dev_perplexity:
            parameters = {'input_emb': model_input_emb.state_dict(), 'enc': model_enc.state_dict(),
                'output_emb': model_output_emb.state_dict(), 'dec': model_dec.state_dict()}
            best_dev_perplexity = dev_perplexity
            torch.save(parameters, get_model_file(args.dataset, args.model_id + "-best"))

        if (epoch - args.saving_from) % args.saving_interval == 0:
            parameters = {'input_emb': model_input_emb.state_dict(), 'enc': model_enc.state_dict(),
                'output_emb': model_output_emb.state_dict(), 'dec': model_dec.state_dict()}
            torch.save(parameters, get_model_file(args.dataset, args.model_id + "-" + str(epoch)))

    parser = Seq2SeqSemanticParser(input_indexer, output_indexer, model_input_emb, model_enc, model_output_emb, model_dec, args)
    return parser

def train_model_rl_warm_start(train_data, test_data, input_indexer, output_indexer, args):
    device = config.device
    if args.warm_model_id is None:
        print("Training warm start model")
        # trickily set hyperparameters
        args_bak = {}
        args_bak["epochs"] = args.epochs

        args.epochs = args.epoch_start
        # do a warm start
        warm_start_train = train_data if args.start_size == 0 else train_data[:args.start_size]
        warm_start_test = test_data if args.start_size == 0 else test_data[:args.start_size]
        decoder = train_model_encdec_ml(warm_start_train, warm_start_test, input_indexer, output_indexer, args)
        parameters = {'input_emb': decoder.model_input_emb.state_dict(), 'enc': decoder.model_enc.state_dict(),
                    'output_emb': decoder.model_output_emb.state_dict(), 'dec': decoder.model_dec.state_dict()}

        torch.save(parameters, get_model_file(args.dataset, args.model_id + "-warm"))
        # restore
        args.epochs = args_bak["epochs"]
        # doing reinfocement learning
        return decoder.model_input_emb, decoder.model_enc, decoder.model_output_emb, decoder.model_dec

    else:
        print("Loading warm start model")

        model_path = get_model_file(args.dataset, args.warm_model_id)
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

        return model_input_emb, model_enc, model_output_emb, model_dec

def monte_carlo_sampling(enc_out_each_word, enc_context_mask,
                            enc_final_states, output_indexer,
                            model_output_emb, model_dec, output_max_len):
    device = config.device
    batch_size = enc_context_mask.size(0)
    sample_size = args.sample_size
    expand_size = batch_size * sample_size
    context_inf_mask = get_inf_mask(enc_context_mask)
    context_inf_mask = context_inf_mask.repeat(sample_size, 1)
    input_words = torch.from_numpy(np.asarray([output_indexer.index_of(SOS_SYMBOL) for _ in range(batch_size)]))
    input_words = input_words.to(device)
    input_words = input_words.unsqueeze(1)
    input_words = input_words.repeat(sample_size, 1)
    enc_out_each_word = enc_out_each_word.repeat(1, sample_size, 1)
    dec_hidden_states = (enc_final_states[0].repeat(1, sample_size, 1), enc_final_states[1].repeat(1, sample_size, 1))
    # expand sample size time

    output_trace = []
    prob_trace = []
    for i in range(output_max_len):
        input_embeded_words = model_output_emb.forward(input_words)
        input_embeded_words = input_embeded_words.reshape((1, expand_size, -1))
        voc_scores, dec_hidden_states = model_dec(input_embeded_words, dec_hidden_states, enc_out_each_word, context_inf_mask)
        output_words = torch.multinomial(voc_scores, 1)
        input_words = output_words.detach()
        output_trace.append(input_words)
        prob_trace.append(torch.gather(voc_scores, 1, input_words))
    # output trace & probtrace : exandsize, 1
    output_trace = torch.cat(output_trace, 1)
    prob_trace = torch.cat(prob_trace, 1)

    return build_output_tokens(output_trace, prob_trace, output_indexer, batch_size, sample_size)

def build_output_tokens(output_trace, prob_trace, output_indexer, batch_size, sample_size):
    EOS = output_indexer.get_index(EOS_SYMBOL)
    acc_log_probs = []
    output_tokens = []
    prob_trace = torch.log(prob_trace)
    for i, trace in enumerate(output_trace.cpu().numpy()):
        toks = []
        acc_prob = 0
        for j, tok_id in enumerate(trace):
            acc_prob += prob_trace[i][j]
            if tok_id == EOS:
                break
            toks.append(tok_id)
        output_tokens.append(toks)
        acc_log_probs.append(acc_prob)
    
    # need B * S
    batch_tokens = [[] for _ in range(batch_size)]
    # for i in range(2):
    cnt = 0
    for i in range(sample_size):
        for j in range(batch_size):
            batch_tokens[j].append(output_tokens[cnt])
            cnt += 1

    acc_log_probs = torch.stack(acc_log_probs).view((sample_size, batch_size)).transpose(0, 1)

    return batch_tokens, acc_log_probs

def naive_beam_sampling(enc_out_each_word, enc_context_mask,
                            enc_final_states, output_indexer,
                            model_output_emb, model_dec, output_max_len):
    batch_size = enc_context_mask.size(0)
    sample_size = args.sample_size

    # target a list of B * sample_size
    # a list of sum log probas B * sample_size
    enc_out_each_word_list = enc_out_each_word.unbind(1)
    enc_context_mask_list = enc_context_mask.unbind(0)
    enc_final_h_list = enc_final_states[0].unbind(1)
    enc_final_c_list = enc_final_states[1].unbind(1)

    batch_tokens = []
    batch_probs = []
    for id_exs in range(batch_size):
        single_tokens, single_probs = batched_beam_sampling(enc_out_each_word_list[id_exs].unsqueeze(1), enc_context_mask_list[id_exs].unsqueeze(0),
            (enc_final_h_list[id_exs].unsqueeze(1), enc_final_c_list[id_exs].unsqueeze(1)), output_indexer, model_output_emb, model_dec, output_max_len, sample_size)

        # ensure correctness
        # if len(batch_tokens) < sample_size:
        for _ in range(len(single_tokens),sample_size):
            single_tokens.append([])
            x = torch.tensor(-1000000.0).to(config.device)
            single_probs.append(x)

        single_probs = torch.stack(single_probs).to(config.device)
        batch_tokens.append(single_tokens)
        batch_probs.append(single_probs)

    batch_probs = torch.stack(batch_probs)
    return batch_tokens, batch_probs

def mml_loss(acc_log_probs, output_rewards):
    reward = np.exp(acc_log_probs.detach().cpu().numpy()) * output_rewards
    reward = reward.mean(1).mean()
    loss = - acc_log_probs * output_rewards
    loss = loss.mean(1).mean()
    return loss, reward

def norm_mml_loss(acc_log_probs, output_rewards):
    reward = np.exp(acc_log_probs.detach().cpu().numpy()) * output_rewards
    reward = reward.mean(1).mean()
    rewards_sum = output_rewards.sum(1, keepdim=True)
    rewards_sum += 1e-7
    output_rewards = output_rewards / rewards_sum
    loss = - acc_log_probs * output_rewards
    loss = loss.sum(1).mean()
    return loss, reward

def origin_mml_loss(acc_log_probs, output_rewards):
    probs = torch.exp(acc_log_probs.detach())
    reward = probs * output_rewards
    reward = reward.mean(1).mean()
    output_rewards = output_rewards * probs
    rewards_sum = output_rewards.sum(1, keepdim=True)
    rewards_sum += 1e-7
    output_rewards = output_rewards / rewards_sum
    loss = - acc_log_probs * output_rewards
    loss = loss.sum(1).mean()
    return loss, reward

def reward_loss(acc_log_probs, output_rewards):
    reward = np.exp(acc_log_probs.detach().cpu().numpy()) * output_rewards
    reward = reward.mean(1).mean()
    output_rewards = output_rewards - output_rewards.mean(1, True)
    loss = - acc_log_probs * output_rewards
    loss = loss.mean(1).mean()
    return loss, reward

def train_decoder_with_oracle(enc_out_each_word, enc_context_mask,
                            enc_final_states, output_indexer, batch_out, batch_ids,
                            model_output_emb, model_dec, output_max_len, split):
    device = config.device
    if args.do_montecarlo:
        output_tokens, acc_log_probs = monte_carlo_sampling(enc_out_each_word, enc_context_mask,
                                enc_final_states, output_indexer,
                                model_output_emb, model_dec, output_max_len)
    else:
        output_tokens, acc_log_probs = naive_beam_sampling(enc_out_each_word, enc_context_mask,
                    enc_final_states, output_indexer, model_output_emb, model_dec, output_max_len)

    if args.oracle_mode == "sketch":
        output_rewards, num_coverage, num_match = parallel_orcale_reward(output_tokens, batch_ids, split, cache, output_indexer)
    else:
        output_rewards, num_coverage, num_match = parallel_dfa_reward(output_tokens, batch_out, split, cache, output_indexer)

    output_rewards = torch.from_numpy(output_rewards).float().to(device)
    loss, reward = origin_mml_loss(acc_log_probs, output_rewards)
    return loss, reward, num_coverage, num_match


def oracle_perplexity(test_loader,
                    model_input_emb, model_enc, model_output_emb, model_dec,
                    input_indexer, output_indexer, output_max_len):
    device = config.device
    model_input_emb.eval()
    model_enc.eval()
    model_output_emb.eval()
    model_dec.eval()

    test_iter = iter(test_loader)
    epoch_coverage = 0
    epoch_match = 0
    epoch_reward = 0.0

    _do_montecarlo = args.do_montecarlo
    args.do_montecarlo = False
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_iter):
            # print("Evaluating {}".format(batch_idx), file=sys.stderr)
            batch_in, batch_in_lens, batch_out, batch_out_lens, batch_ids = batch_data
            batch_in, batch_in_lens, batch_out, batch_out_lens = \
                batch_in.to(device), batch_in_lens.to(device), batch_out.to(device), batch_out_lens.to(device)
            # print(batch_in.size(), batch_in_lens.size())
            enc_out_each_word, enc_context_mask, enc_final_states = \
                encode_input_for_decoder(batch_in, batch_in_lens, model_input_emb, model_enc)

            loss, reward, num_coverage,  num_match = \
                train_decoder_with_oracle(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                batch_out, batch_ids, model_output_emb, model_dec, output_max_len, "val")
            epoch_coverage += num_coverage
            epoch_match += num_match
            epoch_reward += reward
    perperlexity = epoch_match
    args.do_montecarlo = _do_montecarlo
    return -perperlexity

def eval_mode(*args):
    for x in args:
        x.eval()

def train_mode(*args):
    for x in args:
        x.train()

def train_model_encdec_rl(train_data, test_data, input_indexer, output_indexer, args):
    device = config.device
    # set a warm start, train a ml model and start training from that point
    model_input_emb, model_enc, model_output_emb, model_dec = \
        train_model_rl_warm_start(train_data, test_data, input_indexer, output_indexer, args)

    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    train_input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    test_input_max_len =  np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
    input_max_len = max(train_input_max_len, test_input_max_len)
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    train_output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, train_output_max_len)
    test_output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in test_data]))
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer,  test_output_max_len)
    all_test_output_data = np.maximum(all_test_output_data, 0)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))

    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters

    # optimizer = None
    train_loader = BatchDataLoader(train_data, all_train_input_data, all_train_output_data, batch_size=args.batch_size, shuffle=True, return_id=True)
    test_loader = BatchDataLoader(test_data, all_test_input_data, all_test_output_data, batch_size=args.batch_size, shuffle=False, return_id=True)
    train_iter = iter(train_loader)

    optimizer = optim.Adam([
        {'params': model_input_emb.parameters()},
        {'params': model_enc.parameters()},
        {'params': model_output_emb.parameters()},
        {'params': model_dec.parameters()}], lr=0.0003)

    clip = args.clip_grad
    best_dev_perplexity = np.inf
    for epoch in range(1, args.epochs + 1):
        
        train_mode(model_input_emb, model_enc, model_output_emb, model_dec)
        print('epoch {}'.format(epoch))
        epoch_loss = 0.0
        num_batch = 0.0
        epoch_coverage = 0
        epoch_match = 0
        epoch_reward = 0.0
        for batch_idx, batch_data in enumerate(train_iter):
            # print("Training {} {}".format(epoch, batch_idx), file=sys.stderr)
            optimizer.zero_grad()

            batch_in, batch_in_lens, batch_out, batch_out_lens, batch_ids = batch_data
            batch_in, batch_in_lens, batch_out, batch_out_lens = \
                batch_in.to(device), batch_in_lens.to(device), batch_out.to(device), batch_out_lens.to(device)
            enc_out_each_word, enc_context_mask, enc_final_states = \
                encode_input_for_decoder(batch_in, batch_in_lens, model_input_emb, model_enc)

            loss, reward, num_coverage,  num_match = \
                train_decoder_with_oracle(enc_out_each_word, enc_context_mask, enc_final_states, output_indexer,
                batch_out, batch_ids, model_output_emb, model_dec, args.decoder_len_limit, "train")
            epoch_coverage += num_coverage
            epoch_match += num_match
            epoch_reward += reward
            epoch_loss += loss.item()
            loss.backward()
            print('    Batch {}, coverage: {}, match {}, loss {}, reward {}'.format(batch_idx, num_coverage, num_match, loss.item(), reward))
            num_batch += 1
            _ = torch.nn.utils.clip_grad_norm_(model_input_emb.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_enc.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_output_emb.parameters(), clip)
            _ = torch.nn.utils.clip_grad_norm_(model_dec.parameters(), clip)
            optimizer.step()

        print('epoch {}, train coverage: {}, train match {}, train loss {}, train reward {}'.format(epoch, epoch_coverage, epoch_match, (epoch_loss / num_batch), (epoch_reward / num_batch)))

        # if (epoch < args.saving_from) or (args.model_id is None):
        #     continue

        # start saving
        dev_perplexity = oracle_perplexity(test_loader, model_input_emb, model_enc, model_output_emb, model_dec, input_indexer, output_indexer, args.decoder_len_limit)
        print('epoch {} dev loss: {}'.format(epoch, dev_perplexity))

        parameters = {'input_emb': model_input_emb.state_dict(), 'enc': model_enc.state_dict(),
                'output_emb': model_output_emb.state_dict(), 'dec': model_dec.state_dict()}

        torch.save(parameters, get_model_file(args.dataset, args.model_id + "-" + str(epoch)))
        cache.rewrite()
        if dev_perplexity <= best_dev_perplexity:
            best_dev_perplexity = dev_perplexity
            torch.save(parameters, get_model_file(args.dataset, args.model_id + "-best"))

    # parser = Seq2SeqSemanticParser(input_indexer, output_indexer, model_input_emb, model_enc, model_output_emb, model_dec, args)
    # return parser

# Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
# every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
# executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None, show_example=False):
    # e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    # print(pred_derivations)
    # selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    selected_derivs = [x[0] for x in pred_derivations]
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            if show_example:
                print('Example %d' % i)
                print('  x      = "%s"' % ex.x)
                print('  y_tok  = "%s"' % ex.y_tok)
                print('  y_pred = "%s"' % selected_derivs[i].y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i].y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i].y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if  dfa_eual_test(' '.join(ex.y_tok), ' '.join(selected_derivs[i].y_toks)):
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()

if __name__ == '__main__':
    args = _parse_args()
    print(args)
    # global device
    set_global_device(args.gpu)
    if args.do_rl:
        args.oracle_mode = "sketch" if 'Sketch' in args.dataset else 'regex'
        if args.oracle_mode == 'sketch':
            cache = SynthCache(args.cache_id, args.dataset)
        else:
            cache = DFACache(args.cache_id, args.dataset)

    print("Pytroch using device ", config.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data
    train, dev, input_indexer, output_indexer = load_datasets(args.dataset)
    train_data_indexed, dev_data_indexed = index_datasets(train, dev, input_indexer, output_indexer, args.decoder_len_limit)

    print("Original %i train exs, %i dev exs" % (len(train_data_indexed), len(dev_data_indexed)))
    # train_data_indexed = filter_data(train_data_indexed)
    # dev_data_indexed = filter_data(dev_data_indexed)

    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    # print("Input indexer: %s" % input_indexer)
    # print("Output indexer: %s" % output_indexer)
    # print("Here are some examples post tokenization and indexing:")
    # for i in range(0, min(len(train_data_indexed), 10)):
    #     print(train_data_indexed[i])

    try:
        if args.do_rl:
            train_model_encdec_rl(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        else:
            train_model_encdec_ml(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
    except Exception as err:
        print("Exception Catched")
        if args.do_rl:
            cache.rewrite()
        print(err)
        raise err
    except KeyboardInterrupt:
        print("KeyboardInterrupt Catched")
        if args.do_rl:
            cache.rewrite()
    if args.do_rl:
        cache.rewrite()
