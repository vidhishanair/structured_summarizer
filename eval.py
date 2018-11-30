from __future__ import unicode_literals, print_function, division

import os
import time
import sys
import gc

from tensorboardX import SummaryWriter
import tensorflow as tf
import torch

from utils import config
from utils.batcher import Batcher
from utils.data import Vocab

from utils.utils import calc_running_avg_loss
from utils.train_util import get_input_from_batch, get_output_from_batch
from models.model import Model

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        self.summary_writer = SummaryWriter(eval_dir)
        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_token_mask, enc_padding_sent_mask,  enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)

        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_hidden, max_encoder_output = self.model.encoder(enc_batch, enc_sent_lens, enc_doc_lens, enc_padding_token_mask, enc_padding_sent_mask)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                       encoder_outputs, enc_padding_sent_mask, c_t_1,
                                                                                       extra_zeros, enc_batch_extend_vocab,
                                                                                       coverage)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        del enc_batch, enc_padding_token_mask, enc_padding_sent_mask,  enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
        gc.collect()
        torch.cuda.empty_cache()

        return loss.item()

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            # if iter % 100 == 0:
            #     self.summary_writer.flush()
            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
                    iter, print_interval, time.time() - start, running_avg_loss))
                start = time.time()
            batch = self.batcher.next_batch()


if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()


