from torch.autograd import Variable
import numpy as np
import torch
import utils.config as config


def get_input_from_batch(batch, use_cuda, args):
    batch_size = len(batch.enc_doc_lens)
    device = torch.device("cuda" if config.use_gpu else "cpu")

    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_tags_batch = Variable(torch.from_numpy(batch.enc_tags_batch).long())
    word_batch = Variable(torch.from_numpy(batch.enc_word_batch).long())
    word_padding_mask = Variable(torch.from_numpy(batch.enc_padding_word_mask)).float()
    enc_padding_mask = Variable(torch.from_numpy(batch.enc_padding_mask)).float()
    enc_padding_token_mask = Variable(torch.from_numpy(batch.enc_padding_token_mask)).float()
    enc_padding_sent_mask = Variable(torch.from_numpy(batch.enc_padding_sent_mask)).float()
    # enc_lens = batch.enc_lens
    enc_doc_lens = batch.enc_doc_lens
    enc_sent_lens = batch.enc_sent_lens
    enc_word_lens = batch.enc_word_lens

    extra_zeros = None
    enc_batch_extend_vocab = None

    if args.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    # c_t_1 = Variable(torch.zeros((batch_size, 2 * config.hidden_dim)))
    c_t_1 = Variable(torch.zeros((batch_size, 2 * config.sem_dim_size))) # add 4 * for pointergen
    coverage = None
    if args.is_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.to(device)
        enc_tags_batch = enc_tags_batch.to(device)
        enc_padding_mask = enc_padding_mask.to(device)
        enc_padding_sent_mask = enc_padding_sent_mask.to(device)
        enc_padding_token_mask = enc_padding_token_mask.to(device)
        word_batch = word_batch.to(device)
        word_padding_mask = word_padding_mask.to(device)

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
        if extra_zeros is not None:
            extra_zeros = extra_zeros.to(device)
        c_t_1 = c_t_1.to(device)

        if coverage is not None:
            coverage = coverage.to(device)

    return enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch


def get_output_from_batch(batch, use_cuda):
    device = torch.device("cuda" if config.use_gpu else "cpu")

    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    dec_padding_mask = Variable(torch.from_numpy(batch.dec_padding_mask)).float()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()

    if use_cuda:
        dec_batch = dec_batch.to(device)
        dec_padding_mask = dec_padding_mask.to(device)
        dec_lens_var = dec_lens_var.to(device)
        target_batch = target_batch.to(device)

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch
