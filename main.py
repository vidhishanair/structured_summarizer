from __future__ import unicode_literals, print_function, division

import os
import time
import gc
import argparse
import logging
import math
import numpy as np
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
from sklearn.metrics import classification_report

use_cuda = config.use_gpu and torch.cuda.is_available()
#print('Devices available: '+str(torch.cuda.current_device()))
# device = torch.device("cuda" if config.use_gpu else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0,1")

class Train(object):
    def __init__(self, args, model_name=None):
        vocab = args.vocab_path if args.vocab_path is not None else config.vocab_path
        self.vocab = Vocab(vocab, config.vocab_size, config.embeddings_file, args)
        self.train_batcher = Batcher(args.train_data_path, self.vocab, mode='train',
                                     batch_size=args.batch_size, single_pass=False, args=args)
        self.eval_batcher = Batcher(args.eval_data_path, self.vocab, mode='eval',
                                    batch_size=args.batch_size, single_pass=True, args = args)
        time.sleep(30)

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
        self.model = nn.DataParallel(Model(args, self.vocab)).to(device)

        params = list(self.model.module.encoder.parameters()) + list(self.model.module.decoder.parameters()) + \
                 list(self.model.module.reduce_state.parameters())

        initial_lr = args.lr_coverage if args.is_coverage else args.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        self.crossentropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.head_child_crossent = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor([0.1,1]).cuda())
        self.attn_mse_loss = nn.MSELoss()

        start_iter, start_loss = 0, 0

        if args.reload_path is not None:
            print('Loading from checkpoint: '+str(args.reload_path))
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
        loss, _, _, _ = self.get_loss(batch, args)
        if loss is None:
            return None
        s1 = time.time()
        loss.backward()
        #print("time for backward: "+str(time.time() - s1))

        clip_grad_norm(self.model.module.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.module.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm(self.model.module.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()
        return loss.item()

    def train_iters(self, n_iters, args):
        start_iter, running_avg_loss = self.setup_train(args)
        logger = self.setup_logging()
        logger.debug(str(args))
        logger.debug(str(config))

        start = time.time()
        best_val_loss = None

        for it in tqdm(range(n_iters)):
            iter = start_iter + it
            self.model.module.train()
            batch = self.train_batcher.next_batch()
            start1 = time.time()
            loss = self.train_one_batch(batch, args)
            #print("time for 1 batch+get: "+str(time.time() - start))
            #print("time for 1 batch: "+str(time.time() - start1))
            #start=time.time()
            #print(loss)
            # for n,p in self.model.module.encoder.named_parameters():
            #     print('===========\ngradient:{}\n----------\n{}'.format(n,p.grad))
            # exit()
            if math.isnan(loss):
                msg = "Loss has reached NAN. Exiting"
                logger.debug(msg)
                print(msg)
                exit()
            if loss is not None:
                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)
                iter += 1

            print_interval = 200
            if iter % print_interval == 0:
                msg = 'steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss)
                print(msg)
                logger.debug(msg)
                start = time.time()
            if iter % config.eval_interval == 0:
                print("Starting Eval")
                loss = self.run_eval(logger, args)
                if best_val_loss is None or loss < best_val_loss:
                    best_val_loss = loss
                    self.save_model(running_avg_loss, iter)
                    print("Saving best model")
                    logger.debug("Saving best model")

    def get_loss(self, batch, args, mode='train'):

        s2 = time.time()
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, \
            enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, word_batch, word_padding_mask, enc_word_lens, \
                enc_tags_batch, enc_sent_tags, enc_sent_token_mat, adj_mat, weighted_adj_mat, norm_adj_mat,\
                    parent_heads = get_input_from_batch(batch, use_cuda, args)
        #print("time for input func: "+str(time.time() - s2))

        final_dist_list, attn_dist_list, p_gen_list, coverage_list, sent_attention_matrix, \
        sent_single_head_scores, sent_all_head_scores, sent_all_child_scores, \
        token_score, sent_score, doc_score = self.model.forward(enc_batch, enc_padding_token_mask,
                                                                                        enc_padding_sent_mask,
                                                                                        enc_doc_lens,
                                                                                        enc_sent_lens,
                                                                                        enc_batch_extend_vocab,
                                                                                        extra_zeros,
                                                                                        c_t_1, coverage,
                                                                                        word_batch,
                                                                                        word_padding_mask,
                                                                                        enc_word_lens,
                                                                                        enc_tags_batch,
                                                                                        enc_sent_token_mat,
                                                                                        max_dec_len,
                                                                                        dec_batch, adj_mat,
                                                                                        weighted_adj_mat, args)

        step_losses = []
        loss = 0
        ind_losses = {
            'summ_loss':0,
            'sent_single_head_loss':0,
            'sent_all_head_loss':0,
            'sent_all_child_loss':0,
            'token_contsel_loss':0,
            'sent_imp_loss':0,
            'doc_imp_loss':0
        }
        counts = {'token_consel_num_correct' : 0,
                  'token_consel_num' : 0,
                  'sent_imp_num_correct' : 0,
                  'doc_imp_num_correct' : 0,
                  'sent_single_heads_num_correct' : 0,
                  'sent_single_heads_num' : 0,
                  'sent_all_heads_num_correct' : 0,
                  'sent_all_heads_num' : 0,
                  'sent_all_child_num_correct' : 0,
                  'sent_all_child_num' : 0}
        eval_data = {}
        
        s1 = time.time()
        if args.use_summ_loss:
            for di in range(min(max_dec_len, args.max_dec_steps)):
                final_dist = final_dist_list[:, di, :]
                attn_dist = attn_dist_list[:, di, :]
                if args.is_coverage:
                    coverage = coverage_list[:, di, :]

                target = target_batch[:, di]
                gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
                step_loss = -torch.log(gold_probs + config.eps)
                if args.is_coverage:
                    step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                    step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                step_mask = dec_padding_mask[:, di]
                step_loss = step_loss * step_mask
                step_losses.append(step_loss)
            sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
            batch_avg_loss = sum_losses / dec_lens_var
            loss += torch.mean(batch_avg_loss)
            ind_losses['summ_loss'] += torch.mean(batch_avg_loss).item()

        if args.heuristic_chains:
            if args.use_attmat_loss:
                pred = sent_attention_matrix[:,:,1:].contiguous().view(-1)
                gold = norm_adj_mat.view(-1)
                loss_aux = self.attn_mse_loss(pred, gold)
                loss += 100*loss_aux
            if args.use_sent_single_head_loss:
                pred = sent_single_head_scores
                pred = pred.view(-1, pred.size(2))
                head_labels = parent_heads.view(-1)
                loss_aux = self.crossentropy(pred, head_labels.long())
                loss += loss_aux
                prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
                if mode == 'eval':
                    prediction[head_labels==-1] = -2 # Explicitly set masked tokens as different from value in gold
                    counts['sent_single_heads_num_correct'] = torch.sum(prediction.eq(head_labels.long())).item()
                    counts['sent_single_heads_num'] = torch.sum(head_labels != -1).item()
                ind_losses['sent_single_head_loss'] += loss_aux.item()
            if args.use_sent_all_head_loss:
                pred = sent_all_head_scores
                pred = pred.view(-1, pred.size(3))
                target_h = adj_mat.permute(0,2,1).contiguous().view(-1)
                #print(pred.size(), target.size())
                loss_aux = self.head_child_crossent(pred, target_h.long())
                loss += loss_aux
                prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
                if mode == 'eval':
                    prediction[target_h==-1] = -2 # Explicitly set masked tokens as different from value in gold
                    counts['sent_all_heads_num_correct'] = torch.sum(prediction.eq(target_h.long())).item()
                    counts['sent_all_heads_num_correct_1'] = torch.sum(prediction[target_h==1].eq(target_h[target_h==1].long())).item()
                    counts['sent_all_heads_num_correct_0'] = torch.sum(prediction[target_h==0].eq(target_h[target_h==0].long())).item()
                    counts['sent_all_heads_num_1'] = torch.sum(target_h == 1).item()
                    counts['sent_all_heads_num_0'] = torch.sum(target_h == 0).item()
                    counts['sent_all_heads_num'] = torch.sum(target_h != -1).item()
                    eval_data['sent_all_heads_pred'] = prediction.numpy()
                    eval_data['sent_all_heads_true'] = target_h.numpy()
                ind_losses['sent_all_head_loss'] += loss_aux.item()
                #print('all head '+str(loss_aux.item()))
            if args.use_sent_all_child_loss:
                pred = sent_all_child_scores
                pred = pred.view(-1, pred.size(3))
                target = adj_mat.contiguous().view(-1)
                loss_aux = self.head_child_crossent(pred, target.long())
                loss += loss_aux
                prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
                if mode == 'eval':
                    prediction[target==-1] = -2 # Explicitly set masked tokens as different from value in gold
                    counts['sent_all_child_num_correct'] = torch.sum(prediction.eq(target.long())).item()
                    counts['sent_all_child_num_correct_1'] = torch.sum(prediction[target==1].eq(target[target==1].long())).item()
                    counts['sent_all_child_num_correct_0'] = torch.sum(prediction[target==0].eq(target[target==0].long())).item()
                    counts['sent_all_child_num_1'] = torch.sum(target == 1).item()
                    counts['sent_all_child_num_0'] = torch.sum(target == 0).item()
                    counts['sent_all_child_num'] = torch.sum(target != -1).item()
                    eval_data['sent_all_child_pred'] = prediction.numpy()
                    eval_data['sent_all_child_true'] = target.numpy()
                ind_losses['sent_all_child_loss'] += loss_aux.item()
                #print('all child '+str(loss_aux.item()))
            # print(target_h.long().eq(target.long()))
            # print(adj_mat)
            #else:
            #   pass

        if args.use_token_contsel_loss:
            pred = token_score.view(-1, 2)
            gold = enc_tags_batch.view(-1)
            loss1 = self.crossentropy(pred, gold.long())
            loss += loss1
            if mode == 'eval':
                prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
                prediction[gold==-1] = -2 # Explicitly set masked tokens as different from value in gold
                counts['token_consel_num_correct'] = torch.sum(prediction.eq(gold)).item()
                counts['token_consel_num'] = torch.sum(gold != -1).item()
            ind_losses['token_contsel_loss'] += loss1.item()
        if args.use_sent_imp_loss:
            pred = sent_score.view(-1)
            enc_sent_tags[enc_sent_tags == -1] = 0
            gold = enc_sent_tags.sum(dim=-1).float()
            gold = gold / gold.sum(dim=1, keepdim=True).repeat(1, gold.size(1))
            gold = gold.view(-1)
            loss2 = self.attn_mse_loss(pred, gold)
            ind_losses['sent_imp_loss'] += loss2.item()
            loss += loss2
        if args.use_doc_imp_loss:
            pred = doc_score.view(-1)
            count_tags = enc_tags_batch.clone().detach()
            count_tags[count_tags == 0] = 1
            count_tags[count_tags == -1] = 0
            token_count = count_tags.sum(dim=-1).sum(dim=-1)
            enc_tags_batch[enc_tags_batch == -1] = 0
            gold = enc_tags_batch.sum(dim=-1)
            gold = gold.sum(dim=-1)
            gold = gold / token_count
            loss3 = self.attn_mse_loss(pred, gold)
            loss += loss3
            ind_losses['doc_imp_loss'] += loss3.item()
        #print("time for loss compute: "+str(time.time() - s1))
        #print("time for 1 batch func: "+str(time.time() - s2))
        return loss, ind_losses, counts, eval_data

    def run_eval(self, logger, args):
        running_avg_loss, iter = 0, 0
        run_avg_losses = {
            'summ_loss':0,
            'sent_single_head_loss':0,
            'sent_all_head_loss':0,
            'sent_all_child_loss':0,
            'token_contsel_loss':0,
            'sent_imp_loss':0,
            'doc_imp_loss':0
        }
        counts = {'token_consel_num_correct' : 0,
                  'token_consel_num' : 0,
                  'sent_single_heads_num_correct' : 0,
                  'sent_single_heads_num' : 0,
                  'sent_all_heads_num_correct' : 0,
                  'sent_all_heads_num' : 0,
                  'sent_all_heads_num_correct_1' : 0,
                  'sent_all_heads_num_1' : 0,
                  'sent_all_heads_num_correct_0' : 0,
                  'sent_all_heads_num_0' : 0,
                  'sent_all_child_num_correct' : 0,
                  'sent_all_child_num' : 0,
                  'sent_all_child_num_correct_1' : 0,
                  'sent_all_child_num_1' : 0,
                  'sent_all_child_num_correct_0' : 0,
                  'sent_all_child_num_0' : 0}
        eval_res = {'sent_all_heads_pred': [],
                    'sent_all_heads_true': [],
                    'sent_all_child_pred': [],
                    'sent_all_child_true': [],
                    }
        self.model.module.eval()
        self.eval_batcher._finished_reading = False
        self.eval_batcher.setup_queues()
        batch = self.eval_batcher.next_batch()
        while batch is not None:
            loss, sample_ind_losses, sample_counts, eval_data = self.get_loss(batch, args, mode='eval')
            loss = loss.item()
            if loss is not None:
                running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, iter)

                if args.use_summ_loss:
                    run_avg_losses['summ_loss'] = calc_running_avg_loss(sample_ind_losses['summ_loss'], run_avg_losses['summ_loss'], iter)
                if args.use_sent_single_head_loss:
                    run_avg_losses['sent_single_head_loss'] = calc_running_avg_loss(sample_ind_losses['sent_single_head_loss'], run_avg_losses['sent_single_head_loss'], iter)
                    counts['sent_single_heads_num_correct'] += sample_counts['sent_single_heads_num_correct']
                    counts['sent_single_heads_num'] += sample_counts['sent_single_heads_num']
                if args.use_sent_all_head_loss:
                    run_avg_losses['sent_all_head_loss'] = calc_running_avg_loss(sample_ind_losses['sent_all_head_loss'], run_avg_losses['sent_all_head_loss'], iter)
                    counts['sent_all_heads_num_correct'] += sample_counts['sent_all_heads_num_correct']
                    counts['sent_all_heads_num'] += sample_counts['sent_all_heads_num']
                    counts['sent_all_heads_num_correct_1'] += sample_counts['sent_all_heads_num_correct_1']
                    counts['sent_all_heads_num_1'] += sample_counts['sent_all_heads_num_1']
                    counts['sent_all_heads_num_correct_0'] += sample_counts['sent_all_heads_num_correct_0']
                    counts['sent_all_heads_num_0'] += sample_counts['sent_all_heads_num_0']
                    eval_res['sent_all_heads_pred'].append(eval_data['sent_all_heads_pred'])
                    eval_res['sent_all_heads_true'].append(eval_data['sent_all_heads_true'])
                if args.use_sent_all_child_loss:
                    run_avg_losses['sent_all_child_loss'] = calc_running_avg_loss(sample_ind_losses['sent_all_child_loss'], run_avg_losses['sent_all_child_loss'], iter)
                    counts['sent_all_child_num_correct'] += sample_counts['sent_all_child_num_correct']
                    counts['sent_all_child_num'] += sample_counts['sent_all_child_num']
                    counts['sent_all_child_num_correct_1'] += sample_counts['sent_all_child_num_correct_1']
                    counts['sent_all_child_num_1'] += sample_counts['sent_all_child_num_1']
                    counts['sent_all_child_num_correct_0'] += sample_counts['sent_all_child_num_correct_0']
                    counts['sent_all_child_num_0'] += sample_counts['sent_all_child_num_0']
                    eval_res['sent_all_child_pred'].append(eval_data['sent_all_child_pred'])
                    eval_res['sent_all_child_true'].append(eval_data['sent_all_child_true'])
                if args.use_token_contsel_loss:
                    run_avg_losses['token_contsel_loss'] = calc_running_avg_loss(sample_ind_losses['token_contsel_loss'], run_avg_losses['token_contsel_loss'], iter)
                    counts['token_consel_num_correct'] += sample_counts['token_consel_num_correct']
                    counts['token_consel_num'] += sample_counts['token_consel_num']
                if args.use_sent_imp_loss:
                    run_avg_losses['sent_imp_loss'] = calc_running_avg_loss(sample_ind_losses['sent_imp_loss'], run_avg_losses['sent_imp_loss'], iter)
                if args.use_doc_imp_loss:
                    run_avg_losses['doc_imp_loss'] = calc_running_avg_loss(sample_ind_losses['doc_imp_loss'], run_avg_losses['doc_imp_loss'], iter)
                iter += 1
            batch = self.eval_batcher.next_batch()

        msg = 'Eval: loss: %f' % running_avg_loss
        print(msg)
        logger.debug(msg)

        if args.use_summ_loss:
            msg = 'Summ Eval: loss: %f' % run_avg_losses['summ_loss']
            print(msg)
            logger.debug(msg)
        if args.use_sent_single_head_loss:
            msg = 'Single Sent Head Eval: loss: %f' % run_avg_losses['sent_single_head_loss']
            print(msg)
            logger.debug(msg)
            msg = 'Average Sent Single Head Accuracy: %f' % (counts['sent_single_heads_num_correct']/float(counts['sent_single_heads_num']))
            print(msg)
            logger.debug(msg)
        if args.use_sent_all_head_loss:
            msg = 'All Sent Head Eval: loss: %f' % run_avg_losses['sent_all_head_loss']
            print(msg)
            logger.debug(msg)
            msg = 'Average Sent All Head Accuracy: %f' % (counts['sent_all_heads_num_correct']/float(counts['sent_all_heads_num']))
            print(msg)
            logger.debug(msg)
            # msg = 'Average Sent All Head Class1 Accuracy: %f' % (counts['sent_all_heads_num_correct_1']/float(counts['sent_all_heads_num_1']))
            # print(msg)
            # logger.debug(msg)
            # msg = 'Average Sent All Head Class0 Accuracy: %f' % (counts['sent_all_heads_num_correct_0']/float(counts['sent_all_heads_num_0']))
            # print(msg)
            # logger.debug(msg)
            y_pred = np.concatenate(eval_res['sent_all_heads_pred'])
            y_true = np.concatenate(eval_res['sent_all_heads_true'])
            msg = classification_report(y_true, y_pred, labels=[0,1])
            print(msg)
            logger.debug(msg)


        if args.use_sent_all_child_loss:
            msg = 'All Sent Child Eval: loss: %f' % run_avg_losses['sent_all_child_loss']
            print(msg)
            logger.debug(msg)
            msg = 'Average Sent All Child Accuracy: %f' % (counts['sent_all_child_num_correct']/float(counts['sent_all_child_num']))
            print(msg)
            logger.debug(msg)
            # msg = 'Average Sent All Child Class1 Accuracy: %f' % (counts['sent_all_child_num_correct_1']/float(counts['sent_all_child_num_1']))
            # print(msg)
            # logger.debug(msg)
            # msg = 'Average Sent All Child Class0 Accuracy: %f' % (counts['sent_all_child_num_correct_0']/float(counts['sent_all_child_num_0']))
            # print(msg)
            # logger.debug(msg)
            y_pred = np.concatenate(eval_res['sent_all_child_pred'])
            y_true = np.concatenate(eval_res['sent_all_child_true'])
            msg = classification_report(y_true, y_pred, labels=[0,1])
            print(msg)
            logger.debug(msg)
        if args.use_token_contsel_loss:
            msg = 'Token Contsel Eval: loss: %f' % run_avg_losses['token_contsel_loss']
            print(msg)
            logger.debug(msg)
            msg = 'Average token content sel Accuracy: %f' % (counts['token_consel_num_correct']/float(counts['token_consel_num']))
            print(msg)
            logger.debug(msg)
        if args.use_sent_imp_loss:
            msg = 'Sent Imp Eval: loss: %f' % run_avg_losses['sent_imp_loss']
            print(msg)
            logger.debug(msg)
        if args.use_doc_imp_loss:
            msg = 'Doc Imp Eval: loss: %f' % run_avg_losses['doc_imp_loss']
            print(msg)
            logger.debug(msg)

        return running_avg_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Structured Summarization Model')
    parser.add_argument('--save_path', type=str, default=None, help='location of the save path')
    parser.add_argument('--reload_path', type=str, default=None, help='location of the older saved path')
    parser.add_argument('--reload_pretrained_clf_path', type=str, default=None, help='location of the older saved path')
    parser.add_argument('--train_data_path', type=str, default='/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/chunked/train_*', help='location of the train data path')
    parser.add_argument('--eval_data_path', type=str, default='/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/val.bin', help='location of the eval data path')
    parser.add_argument('--vocab_path', type=str, default=None, help='location of the eval data path')
    # parser.add_argument('--train_data_path', type=str, default=None, help='location of the train data path')

    #Summ Decoding args
    parser.add_argument('--pointer_gen', action='store_true', default=False, help='use pointer-generator')
    parser.add_argument('--is_coverage', action='store_true', default=False, help='use coverage loss')

    #SA encoder decoder args
    parser.add_argument('--L1_structure_penalty', action='store_true', default=False, help='L2 regularization on Structures')
    parser.add_argument('--sep_sent_features', action='store_true', default=False, help='use sent features for decoding attention')
    parser.add_argument('--token_scores', action='store_true', default=False, help='use token scores for decoding attention')
    parser.add_argument('--sent_scores', action='store_true', default=False, help='use sent scores for decoding attention')
    parser.add_argument('--fixed_scorer', action='store_true', default=False, help='use fixed pretrained scorer')
    parser.add_argument('--use_glove', action='store_true', default=False, help='use_glove_embeddings for training')

    #Pretraining and loss args
    parser.add_argument('--heuristic_chains', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--sm_ner_model', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_ner', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_coref', action='store_true', default=False, help='heuristic coref for training')
    parser.add_argument('--use_summ_loss', action='store_true', default=False, help='use summ loss for training')
    parser.add_argument('--use_token_contsel_loss', action='store_true', default=False, help='use token_level content selection for pre-training')
    parser.add_argument('--use_sent_imp_loss', action='store_true', default=False, help='use sent_level content selection for pre-training')
    parser.add_argument('--use_doc_imp_loss', action='store_true', default=False, help='use doc_level content selection for pre-training')
    parser.add_argument('--use_attmat_loss', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_sent_single_head_loss', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_sent_all_head_loss', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_sent_all_child_loss', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_all_sent_head_at_decode', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_all_sent_child_at_decode', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_single_sent_head_at_decode', action='store_true', default=False, help='decode summarization')

    
    parser.add_argument('--predict_sent_single_head', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_sent_all_head', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_sent_all_child', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_contsel_tags', action='store_true', default=False, help='decode summarization')

    parser.add_argument('--lr', type=float, default=0.15, help='Learning Rate')
    parser.add_argument('--lr_coverage', type=float, default=0.15, help='Learning Rate for Coverage')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--max_dec_steps', type=int, default=100, help='Max Dec Steps')


    # if all false - summarization with just plain attention over sentences - 17.6 or so rouge

    args = parser.parse_args()
    save_path = args.save_path
    # reload_path = args.reload_path

    train_processor = Train(args, save_path)
    train_processor.train_iters(config.max_iterations, args)
