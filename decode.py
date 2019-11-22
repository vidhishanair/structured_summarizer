#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys

#reload(sys)
#sys.setdefaultencoding('utf8')

import os
import time
import argparse

import torch
from torch.autograd import Variable
from dependency_decoding import chu_liu_edmonds
import numpy as np

from utils.batcher import Batcher
from utils.data import Vocab
from utils import data, config
from models.model import Model
from utils.utils import write_for_rouge, rouge_eval, rouge_log, write_to_json_file, write_tags
from utils.train_util import get_input_from_batch, get_output_from_batch
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.coco import COCO


use_cuda = config.use_gpu and torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def extend(self, token, log_prob, state, context, coverage):
        return Beam(tokens = self.tokens + [token],
                    log_probs = self.log_probs + [log_prob],
                    state = state,
                    context = context,
                    coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, args, model_file_path, save_path):
        model_name = os.path.basename(model_file_path)
        self.args= args
        self._decode_dir = os.path.join(config.log_root, save_path, 'decode_%s' % (model_name))
        self._structures_dir = os.path.join(self._decode_dir, 'structures')
        self._sent_heads_dir = os.path.join(self._decode_dir, 'sent_heads_preds')
        self._sent_heads_ref_dir = os.path.join(self._decode_dir, 'sent_heads_ref')
        self._contsel_dir = os.path.join(self._decode_dir, 'content_sel_preds')
        self._contsel_ref_dir = os.path.join(self._decode_dir, 'content_sel_ref')
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')

        self._rouge_ref_file = os.path.join(self._decode_dir, 'rouge_ref.json')
        self._rouge_pred_file = os.path.join(self._decode_dir, 'rouge_pred.json')
        self.stat_res_file = os.path.join(self._decode_dir, 'stats.txt')
        for p in [self._decode_dir, self._structures_dir, self._sent_heads_ref_dir, self._sent_heads_dir, self._contsel_ref_dir,
                self._contsel_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        vocab = args.vocab_path if args.vocab_path is not None else config.vocab_path
        self.vocab = Vocab(vocab, config.vocab_size, config.embeddings_file, args)
        self.batcher = Batcher(args.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True, args=args)
        self.batcher.setup_queues()
        #time.sleep(15)

        self.model = Model(args, self.vocab).to(device)
        self.model.eval()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def extract_structures(self, batch, sent_attention_matrix, doc_attention_matrix, count, use_cuda, sent_scores):
        fileName = os.path.join(self._structures_dir, "%06d_struct.txt" % count)
        fp = open(fileName, "w")
        fp.write("Doc: "+str(count)+"\n")
        #exit(0)
        doc_attention_matrix = doc_attention_matrix[:,:] #this change yet to be tested!
        l = batch.enc_doc_lens[0].item()
        doc_sent_no = 0
        # for i in range(l):
        #     printstr = ''
        #     sent = batch.enc_batch[0][i]
        #     #scores = str_scores_sent[sent_no][0:l, 0:l]
        #     token_count = 0
        #     for j in range(batch.enc_sent_lens[0][i].item()):
        #         token = sent[j].item()
        #         printstr += self.vocab.id2word(token)+" "
        #         token_count = token_count + 1
        #     #print(printstr)
        #     fp.write(printstr+"\n")
        #
        #     scores = sent_attention_matrix[doc_sent_no][0:token_count, 0:token_count]
        #     shape2 = sent_attention_matrix[doc_sent_no][0:token_count,0:token_count].size()
        #     row = torch.ones([1, shape2[1]+1]).cuda()
        #     column = torch.zeros([shape2[0], 1]).cuda()
        #     new_scores = torch.cat([column, scores], dim=1)
        #     new_scores = torch.cat([row, new_scores], dim=0)
        #
        #     heads, tree_score = chu_liu_edmonds(new_scores.data.cpu().numpy().astype(np.float64))
        #     #print(heads, tree_score)
        #     fp.write(str(heads)+" ")
        #     fp.write(str(tree_score)+"\n")
        #     doc_sent_no+=1

        shape2 = doc_attention_matrix[0:l,0:l+1].size()
        row = torch.zeros([1, shape2[1]]).cuda()
        #column = torch.zeros([shape2[0], 1]).cuda()
        scores = doc_attention_matrix[0:l, 0:l+1]
        #new_scores = torch.cat([column, scores], dim=1)
        new_scores = torch.cat([row, scores], dim=0)
        val, root_edge = torch.max(new_scores[:,0], dim=0)
        root_score = torch.zeros([shape2[0]+1,1]).cuda()
        root_score[root_edge] = 1
        new_scores[:,0] = root_score.squeeze()
        #print(new_scores)
        #print(new_scores.sum(dim=0))
        #print(new_scores.sum(dim=1))
        #print(new_scores.size())
        heads, tree_score = chu_liu_edmonds(new_scores.data.cpu().numpy().astype(np.float64))
        #print(heads, tree_score)
        fp.write("\n")
        sentences = str(batch.original_articles[0]).split("<split1>")
        for idx, sent in enumerate(sentences):
            fp.write(str(idx)+"\t"+str(sent)+"\n")
        #fp.write(str("\n".join(batch.original_articles[0].split("<split1>"))+"\n")
        fp.write(str(heads)+" ")
        fp.write(str(tree_score)+"\n")
        s = sent_scores[0].data.cpu().numpy()
        for val in s:
            fp.write(str(val))
        fp.close()
        #exit()

    def decode(self):
        start = time.time()
        counter = 0
        abstract_ref = []
        abstract_pred = []
        batch = self.batcher.next_batch()
        token_contsel_tot_correct, sent_heads_tot_correct = 0,0
        token_contsel_tot_num, sent_heads_tot_num = 0,0
        while batch is not None:
            # Run beam search to get best Hypothesis
            has_summary, best_summary, token_consel_prediction, token_consel_num_correct, token_consel_num, \
            sent_heads_prediction, sent_heads_num_correct, sent_heads_num = self.get_decoded_outputs(batch, counter)
            token_contsel_tot_correct += token_consel_num_correct
            token_contsel_tot_num += token_consel_num
            sent_heads_tot_correct += sent_heads_num_correct
            sent_heads_tot_num += sent_heads_num

            if args.predict_sent_heads:
                no_sents = batch.enc_doc_lens[0]
                prediction = sent_heads_prediction[0:no_sents].tolist()
                ref = batch.original_parent_heads[0]
                write_tags(prediction, ref, counter, self._sent_heads_dir, self._sent_heads_ref_dir)

            if args.predict_contsel_tags:
                no_words = batch.enc_word_lens[0]
                prediction = sent_heads_prediction[0:no_words]
                ref = batch.contsel_tags[0]
                write_tags(prediction, ref, counter, self._contsel_dir, self._contsel_ref_dir)

            if has_summary == False:
                batch = self.batcher.next_batch()
                continue
            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if self.args.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = batch.original_abstracts_sents[0]

            abstract_ref.append(" ".join(original_abstract_sents))
            abstract_pred.append(" ".join(decoded_words))
            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

            batch = self.batcher.next_batch()
            # if counter == 5:
            #    break

        print("Decoder has finished reading dataset for single_pass.")

        fp = open(self.stat_res_file, 'w')
        if args.predict_contsel_tags:
            fp.write("Avg token_contsel: "+str((token_contsel_tot_correct/float(token_contsel_tot_num))))
        if args.predict_sent_heads:
            fp.write("Avg sent heads: "+str((sent_heads_tot_correct/float(sent_heads_tot_num))))
        fp.close()

        #results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        #rouge_log(results_dict, self._decode_dir)

        write_to_json_file(abstract_ref, self._rouge_ref_file)
        write_to_json_file(abstract_pred, self._rouge_pred_file)

        # cocoEval = COCOEvalCap(self._rouge_ref_file, self._rouge_pred_file)
        # cocoEval.evaluate()
        # for metric, score in cocoEval.eval.items():
        #     print('%s: %.3f'%(metric, score))

    def get_decoded_outputs(self, batch, count):
        #batch should have only one example
        enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, \
                        enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0, word_batch, word_padding_mask, enc_word_lens, \
                                enc_tags_batch, enc_sent_tags, enc_sent_token_mat, sup_adj_mat, parent_heads = get_input_from_batch(batch, use_cuda, self.args)

        # if(enc_batch.size()[1]==1 or enc_batch.size()[2]==1): # test why this?
        #     return False, None

        encoder_output = self.model.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask,
                                                         enc_padding_sent_mask, word_batch, word_padding_mask,
                                                         enc_word_lens, enc_tags_batch, enc_sent_token_mat)
        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, \
        enc_batch_extend_vocab, token_level_sentence_scores, sent_outputs, token_scores, sent_scores, sent_matrix = \
                                    self.model.get_app_outputs(encoder_output, enc_padding_token_mask,
                                                   enc_padding_sent_mask, enc_batch_extend_vocab, enc_sent_token_mat)

        mask = enc_padding_sent_mask[0].unsqueeze(0).repeat(enc_padding_sent_mask.size(1),1) * enc_padding_sent_mask[0].unsqueeze(1).transpose(1,0)

        mask = torch.cat((enc_padding_sent_mask[0].unsqueeze(1), mask), dim=1)
        mat = encoder_output['sent_attention_matrix'][0][:,:] * mask

        self.extract_structures(batch, encoder_output['token_attention_matrix'], mat, count, use_cuda, encoder_output['sent_score'])

        token_consel_num_correct, sent_heads_num_correct = 0, 0
        token_consel_num, sent_heads_num = 0, 0
        token_contsel_prediction, sent_heads_prediction = None, None
        if args.predict_contsel_tags:
            pred = encoder_output['token_score'][0, :, :].view(-1, 2)
            token_contsel_gold = enc_tags_batch[0, :].view(-1)
            token_contsel_prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
            token_contsel_prediction[token_contsel_gold==-1] = -2 # Explicitly set masked tokens as different from value in gold
            token_consel_num_correct = torch.sum(token_contsel_prediction.eq(token_contsel_gold)).item()
            token_consel_num = torch.sum(token_contsel_gold != -1).item()

        if args.predict_sent_heads:
            pred = encoder_output['sent_head_scores'][0, :, :]
            # pred = pred.view(-1, pred.size(2))
            head_labels = parent_heads[0, :].view(-1)
            sent_heads_prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
            sent_heads_prediction[head_labels==-1] = -2 # Explicitly set masked tokens as different from value in gold
            sent_heads_num_correct = torch.sum(sent_heads_prediction.eq(head_labels)).item()
            sent_heads_num = torch.sum(head_labels != -1).item()

        results = []
        steps = 0
        has_summary = False
        beams_sorted = [None]
        if args.predict_summaries:
            has_summary = True

            if(args.fixed_scorer):
                scorer_output = self.model.module.pretrained_scorer.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask, enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch)
                token_scores = scorer_output['token_score']
                sent_scores = scorer_output['sent_score'].unsqueeze(1).repeat(1, enc_padding_token_mask.size(2),1, 1).view(enc_padding_token_mask.size(0), enc_padding_token_mask.size(1)*enc_padding_token_mask.size(2))

            s_t_0 = self.model.reduce_state(encoder_last_hidden)

            if config.use_maxpool_init_ctx:
                c_t_0 = max_encoder_output

            dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            #decoder batch preparation, it has beam_size example initially everything is repeated
            beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                          log_probs=[0.0],
                          state=(dec_h[0], dec_c[0]),
                          context = c_t_0[0],
                          coverage=(coverage_t_0[0] if self.args.is_coverage else None))
                     for _ in range(config.beam_size)]

            while steps < config.max_dec_steps and len(results) < config.beam_size:
                latest_tokens = [h.latest_token for h in beams]
                latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                                 for t in latest_tokens]
                y_t_1 = Variable(torch.LongTensor(latest_tokens))
                if use_cuda:
                    y_t_1 = y_t_1.cuda()
                all_state_h =[]
                all_state_c = []

                all_context = []

                for h in beams:
                    state_h, state_c = h.state
                    all_state_h.append(state_h)
                    all_state_c.append(state_c)

                    all_context.append(h.context)

                s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
                c_t_1 = torch.stack(all_context, 0)

                coverage_t_1 = None
                if self.args.is_coverage:
                    all_coverage = []
                    for h in beams:
                        all_coverage.append(h.coverage)
                    coverage_t_1 = torch.stack(all_coverage, 0)

                final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                                                        encoder_outputs, word_padding_mask, c_t_1,
                                                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1, token_scores, sent_scores, sent_outputs)

                topk_log_probs, topk_ids = torch.topk(final_dist, config.beam_size * 2)

                dec_h, dec_c = s_t
                dec_h = dec_h.squeeze()
                dec_c = dec_c.squeeze()

                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t[i]
                    coverage_i = (coverage_t[i] if self.args.is_coverage else None)

                    for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                        new_beam = h.extend(token=topk_ids[i, j].item(),
                                            log_prob=topk_log_probs[i, j].item(),
                                            state=state_i,
                                            context=context_i,
                                            coverage=coverage_i)
                        all_beams.append(new_beam)

                beams = []
                for h in self.sort_beams(all_beams):
                    if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                        if steps >= config.min_dec_steps:
                            results.append(h)
                    else:
                        beams.append(h)
                    if len(beams) == config.beam_size or len(results) == config.beam_size:
                        break

                steps += 1

            if len(results) == 0:
                results = beams

            beams_sorted = self.sort_beams(results)

        return has_summary, beams_sorted[0], \
               token_contsel_prediction, token_consel_num_correct, token_consel_num, \
               sent_heads_prediction, sent_heads_num_correct, sent_heads_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Structured Summarization Model')
    parser.add_argument('--save_path', type=str, default=None, help='location of the save path')
    parser.add_argument('--reload_path', type=str, default=None, help='location of the older saved path')
    parser.add_argument('--decode_data_path', type=str, default='/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/test.bin', help='location of the decode data path')
    parser.add_argument('--vocab_path', type=str, default=None, help='location of the eval data path')


    parser.add_argument('--pointer_gen', action='store_true', default=False, help='use pointer-generator')
    parser.add_argument('--is_coverage', action='store_true', default=False, help='use coverage loss')
    parser.add_argument('--autoencode', action='store_true', default=False, help='use autoencoder setting')
    parser.add_argument('--reload_pretrained_clf_path', type=str, default=None, help='location of the older saved path')

    parser.add_argument('--sep_sent_features', action='store_true', default=False, help='use sent features for decoding attention')
    parser.add_argument('--token_scores', action='store_true', default=False, help='use token scores for decoding attention')
    parser.add_argument('--sent_scores', action='store_true', default=False, help='use sent scores for decoding attention')
    parser.add_argument('--fixed_scorer', action='store_true', default=False, help='use fixed pretrained scorer')
    parser.add_argument('--heuristic_chains', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--link_id_typed', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--max_dec_steps', type=int, default=100, help='Max Dec Steps')
    parser.add_argument('--use_glove', action='store_true', default=False, help='use_glove_embeddings for training')

    parser.add_argument('--predict_summaries', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_sent_heads', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_contsel_tags', action='store_true', default=False, help='decode summarization')



    args = parser.parse_args()
    model_filename = args.reload_path
    save_path = args.save_path
    beam_Search_processor = BeamSearch(args, model_filename, save_path)
    beam_Search_processor.decode()


