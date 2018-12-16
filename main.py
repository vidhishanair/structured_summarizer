from __future__ import unicode_literals, print_function, division

import os
import time
import gc 

from tensorboardX import SummaryWriter
import torch
from models.model import Model
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm

from utils.custom_adagrad import AdagradCustom

from utils import config
from utils.batcher import Batcher
from utils.data import Vocab
from utils.utils import calc_running_avg_loss
from utils.train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.train_batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        self.eval_batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = SummaryWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):

        self.optimizer.zero_grad()
        loss = self.get_loss(batch)
        loss.backward()

        clip_grad_norm(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item()

    def train_iters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        #while iter < n_iters:
        #for iter in tqdm(range(n_iters)):
        best_val_loss = None

        for iter in tqdm(range(n_iters)):
            self.model.train()
            batch = self.train_batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            print_interval = 1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % config.eval_interval == 0:
                loss = self.run_eval()
                if best_val_loss is None or loss < best_val_loss:
                    best_val_loss = loss
                    self.save_model(running_avg_loss, iter)
                    print("Saving best model")

    def get_loss(self, batch):
        enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)
        encoder_outputs, encoder_hidden, max_encoder_output = self.model.encoder(enc_batch, enc_sent_lens, enc_doc_lens,
                                                                                 enc_padding_token_mask,
                                                                                 enc_padding_sent_mask)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self.model.decoder(y_t_1, s_t_1,
                                                                                      encoder_outputs,
                                                                                      enc_padding_sent_mask, c_t_1,
                                                                                      extra_zeros,
                                                                                      enc_batch_extend_vocab,
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
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        del enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        self.model.eval()
        batch = self.eval_batcher.next_batch()
        while batch is not None:
            loss = self.get_loss(batch).item()
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1
            batch = self.eval_batcher.next_batch()
        print('Eval: loss: %f' % running_avg_loss)
        return running_avg_loss


if __name__ == '__main__':
    train_processor = Train()
    train_processor.train_iters(config.max_iterations)
