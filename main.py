from __future__ import unicode_literals, print_function, division

import os
import time
import gc
import argparse
import logging
import math

# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
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
#print('Devices available: '+str(torch.cuda.current_device()))
device = torch.device("cuda" if config.use_gpu else "cpu")

class Train(object):
    def __init__(self, args, model_name=None):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.train_batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                                     batch_size=config.batch_size, single_pass=False, args=args)
        self.eval_batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                                    batch_size=config.batch_size, single_pass=True, args = args)
        time.sleep(15)

        if model_name is None:
            self.train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        else:
            self.train_dir = os.path.join(config.log_root, model_name)

        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)
        self.model_dir = os.path.join(self.train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        #self.summary_writer = SummaryWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.module.encoder.state_dict(),
            'decoder_state_dict': self.model.module.decoder.state_dict(),
            'reduce_state_dict': self.model.module.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, args):
        gpu_ids = [0,1]
        self.model = nn.DataParallel(Model(args), device_ids=gpu_ids).to(device)

        params = list(self.model.module.encoder.parameters()) + list(self.model.module.decoder.parameters()) + \
                 list(self.model.module.reduce_state.parameters())

        initial_lr = config.lr_coverage if args.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        self.sent_crossentropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.attn_mse_loss = nn.MSELoss()

        start_iter, start_loss = 0, 0

        if args.reload_path is not None:
            state = torch.load(args.reload_path, map_location=lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not args.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(device)

        return start_iter, start_loss

    def setup_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        filename = os.path.join(self.train_dir, 'train.log')
        ah = logging.FileHandler(filename)
        ah.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ah.setFormatter(formatter)
        logger.addHandler(ah)
        return logger

    def train_one_batch(self, batch, args):

        self.optimizer.zero_grad()
        self.model.module.encoder.document_structure_att.output = None
        loss = self.get_loss(batch, args)
        if loss is None:
            return None
        loss.backward()

        clip_grad_norm(self.model.module.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.module.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.module.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item()

    def train_iters(self, n_iters, args):
        iter, running_avg_loss = self.setup_train(args)
        logger = self.setup_logging()
        logger.debug(str(args))
        logger.debug(str(config))

        start = time.time()
        best_val_loss = None

        for iter in tqdm(range(n_iters)):
            self.model.module.train()
            batch = self.train_batcher.next_batch()
            loss = self.train_one_batch(batch, args)
            #print(loss)
            if math.isnan(loss):
                exit()
            if loss is not None:
                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
                iter += 1

            print_interval = 1000
            if iter % print_interval == 0:
                msg = 'steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss)
                print(msg)
                logger.debug(msg)
                start = time.time()
                #exit()
            if iter % config.eval_interval == 0:
                loss = self.run_eval(logger, args)
                if best_val_loss is None or loss < best_val_loss:
                    best_val_loss = loss
                    self.save_model(running_avg_loss, iter)
                    print("Saving best model")
                    logger.debug("Saving best model")

    def get_app_outputs(self, encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab):
        encoder_outputs = encoder_output["encoded_tokens"]
        enc_padding_mask = enc_padding_token_mask.contiguous().view(enc_padding_token_mask.size(0),
                                                                        enc_padding_token_mask.size(
                                                                            1) * enc_padding_token_mask.size(2))
        enc_batch_extend_vocab = enc_batch_extend_vocab.contiguous().view(enc_batch_extend_vocab.size(0),
                                                                              enc_batch_extend_vocab.size(
                                                                                  1) * enc_batch_extend_vocab.size(2))
        # else:
        #     encoder_outputs = encoder_output["encoded_sents"]
        #     enc_padding_mask = enc_padding_sent_mask
        encoder_hidden = encoder_output["sent_hidden"]
        max_encoder_output = encoder_output["document_rep"]
        token_level_sentence_scores = encoder_output["token_level_sentence_scores"]
        sent_prediction = encoder_output["sent_prediction"]
        sent_output = encoder_output['encoded_sents']
        return encoder_outputs, enc_padding_mask, encoder_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_prediction, sent_output

    def get_loss(self, batch, args):
        enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, \
            enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch\
            = get_input_from_batch(batch, use_cuda, args)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_output = self.model.module.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask, enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch)
        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_prediction, sent_outputs = \
            self.get_app_outputs(encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab)

        s_t_1 = self.model.module.reduce_state(encoder_last_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output

        step_losses = []
        #print(encoder_outputs)
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self.model.module.decoder.forward(y_t_1, s_t_1,
                                                                                      encoder_outputs,
                                                                                      enc_padding_mask, c_t_1,
                                                                                      extra_zeros,
                                                                                      enc_batch_extend_vocab,
                                                                                      coverage, token_level_sentence_scores, sent_outputs)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            #print(final_dist[-1,:])
            #print(target[-1])
            #print(gold_probs)
            step_loss = -torch.log(gold_probs + config.eps)
            if args.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            #print(step_loss)
            step_losses.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        #print(sum_losses)
        batch_avg_loss = sum_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        #print('Loss: ', loss.item())
        if args.sp_tag_loss:
            pred = sent_prediction.view(-1, 2)
            enc_tags_batch[enc_tags_batch == -1] = 0
            gold = enc_tags_batch.sum(dim=-1)
            gold[gold < 3] = 0
            gold[gold > 0] = 1
            loss_aux = self.sent_crossentropy.forward(pred, gold.view(-1).long())
            #print(loss_aux)
            loss += loss_aux

        if args.tag_norm_loss:
            #sentence_importance_vector = encoder_output['sent_attention_matrix'][:,:,1:].sum(dim=1) * enc_padding_sent_mask
            #sentence_importance_vector = sentence_importance_vector / sentence_importance_vector.sum(dim=1, keepdim=True).repeat(1, sentence_importance_vector.size(1))
            pred = encoder_output['sent_importance_vector'].view(-1)
            enc_tags_batch[enc_tags_batch == -1] = 0
            gold = enc_tags_batch.sum(dim=-1)
            gold = gold / gold.sum(dim=1, keepdim=True).repeat(1, gold.size(1))
            gold = gold.view(-1)
            loss_aux = self.attn_mse_loss(pred, gold)
            #print(loss_aux)
            #print('Aux loss ', (10*loss_aux).item())
            loss += 10*loss_aux
        #if math.isnan(loss.item()):
            #print(encoder_outputs)
        if args.L1_structure_penalty:
            all_linear1_params = torch.cat([x.view(-1) for x in self.model.module.encoder.document_structure_att.output])
            all_linear2_params = torch.cat([x.view(-1) for x in self.model.module.encoder.document_structure_att.output])
            l1_regularization = 0.001 * torch.norm(all_linear1_params, 1)
            l2_regularization = 0.001 * torch.norm(all_linear2_params, 2)
            loss += l1_regularization
        #print(loss)
        del enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, word_batch, word_padding_mask, enc_word_lens
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    def run_eval(self, logger, args):
        running_avg_loss, iter = 0, 0
        self.model.module.eval()
        self.eval_batcher._finished_reading = False
        self.eval_batcher.setup_queues()
        batch = self.eval_batcher.next_batch()
        while batch is not None:
            loss = self.get_loss(batch, args).item()
            if loss is not None:
                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
                iter += 1
            batch = self.eval_batcher.next_batch()
        msg = 'Eval: loss: %f' % running_avg_loss
        print(msg)
        logger.debug(msg)
        return running_avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Structured Summarization Model')
    parser.add_argument('--save_path', type=str, default=None, help='location of the save path')
    parser.add_argument('--reload_path', type=str, default=None, help='location of the older saved path')

    parser.add_argument('--pointer_gen', action='store_true', default=False, help='use pointer-generator')
    parser.add_argument('--is_coverage', action='store_true', default=False, help='use coverage loss')
    parser.add_argument('--autoencode', action='store_true', default=False, help='use autoencoder setting')
    parser.add_argument('--concat_rep', action='store_true', default=False, help='concatenate representation')
    parser.add_argument('--no_sent_sa', action='store_true', default=False, help='no sent SA')
    parser.add_argument('--no_sa', action='store_true', default=False, help='no SA - default encoder')
    parser.add_argument('--sent_score_decoder', action='store_true', default=False, help='add sentence scoring to decoder attentions')
    parser.add_argument('--L1_structure_penalty', action='store_true', default=False, help='L2 regularization on Structures')
    parser.add_argument('--sp_tag_loss', action='store_true', default=False, help='use loss from tags')
    parser.add_argument('--tag_norm_loss', action='store_true', default=False, help='use MSE norm loss from tags')
    parser.add_argument('--gold_tag_scores', action='store_true', default=False, help='use gold tags for scores')
    parser.add_argument('--decode_setting', action='store_true', default=False, help='use gold tags for scores')
    parser.add_argument('--sep_sent_features', action='store_true', default=False, help='use sent features for decoding attention')

    # if all false - summarization with just plain attention over sentences - 17.6 or so rouge

    args = parser.parse_args()
    save_path = args.save_path
    # reload_path = args.reload_path

    train_processor = Train(args, save_path)
    train_processor.train_iters(config.max_iterations, args)
