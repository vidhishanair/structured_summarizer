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
from collections import Counter

from summary_analysis import get_sent_dist
from tree_analysis import find_height, leaf_node_proportion, tree_distance
from utils.batcher import Batcher
from utils.data import Vocab
from utils import data, config
from models.model import Model
from utils.utils import write_for_rouge, rouge_eval, rouge_log, write_to_json_file, write_tags
from utils.train_util import get_input_from_batch, get_output_from_batch
# from pycocoevalcap.eval import COCOEvalCap
# from pycocoevalcap.coco import COCO


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
        self._sent_single_heads_dir = os.path.join(self._decode_dir, 'sent_heads_preds')
        self._sent_single_heads_ref_dir = os.path.join(self._decode_dir, 'sent_heads_ref')
        self._contsel_dir = os.path.join(self._decode_dir, 'content_sel_preds')
        self._contsel_ref_dir = os.path.join(self._decode_dir, 'content_sel_ref')
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')

        self._rouge_ref_file = os.path.join(self._decode_dir, 'rouge_ref.json')
        self._rouge_pred_file = os.path.join(self._decode_dir, 'rouge_pred.json')
        self.stat_res_file = os.path.join(self._decode_dir, 'stats.txt')
        self.sent_count_file = os.path.join(self._decode_dir, 'sent_used_counts.txt')
        for p in [self._decode_dir, self._structures_dir, self._sent_single_heads_ref_dir, self._sent_single_heads_dir, self._contsel_ref_dir,
                self._contsel_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        vocab = args.vocab_path if args.vocab_path is not None else config.vocab_path
        self.vocab = Vocab(vocab, config.vocab_size, config.embeddings_file, args)
        self.batcher = Batcher(args.decode_data_path, self.vocab, mode='decode',
                               batch_size=args.beam_size, single_pass=True, args=args)
        self.batcher.setup_queues()
        time.sleep(30)

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
        height = find_height(heads)
        leaf_nodes = leaf_node_proportion(heads)
        #print(heads, tree_score)
        fp.write("\n")
        sentences = str(batch.original_articles[0]).split("<split1>")
        for idx, sent in enumerate(sentences):
            fp.write(str(idx)+"\t"+str(sent)+"\n")
        #fp.write(str("\n".join(batch.original_articles[0].split("<split1>"))+"\n")
        fp.write(str(heads)+" ")
        fp.write(str(tree_score)+"\n")
        fp.write(str(height)+"\n")
        s = sent_scores[0].data.cpu().numpy()
        for val in s:
            fp.write(str(val))
        fp.close()
        #exit()
        structure_info = dict()
        structure_info['heads'] = heads
        structure_info['height'] = height
        structure_info['leaf_nodes'] = leaf_nodes
        return structure_info

    def decode(self):
        start = time.time()
        counter = 0
        sent_counter = []
        avg_max_seq_len_list = []
        copied_sequence_len = Counter()
        copied_sequence_per_sent = []
        article_copy_id_count_tot = Counter()
        sentence_copy_id_count = Counter()
        novel_counter = Counter()
        repeated_counter = Counter()
        summary_sent_count = Counter()
        summary_sent = []
        article_sent = []
        summary_len = []
        abstract_ref = []
        abstract_pred = []
        sentence_count = []
        tot_sentence_id_count = Counter()
        height_avg = []
        leaf_node_proportion_avg = []
        precision_tree_dist = []
        recall_tree_dist = []
        batch = self.batcher.next_batch()
        height_counter = Counter()
        leaf_nodes_counter = Counter()
        sent_count_fp = open(self.sent_count_file, 'w')


        counts = {'token_consel_num_correct' : 0,
                  'token_consel_num' : 0,
                  'sent_single_heads_num_correct' : 0,
                  'sent_single_heads_num' : 0,
                  'sent_all_heads_num_correct' : 0,
                  'sent_all_heads_num' : 0,
                  'sent_all_child_num_correct' : 0,
                  'sent_all_child_num' : 0}
        no_batches_processed = 0
        while batch is not None:
            # Run beam search to get best Hypothesis
            #start = time.process_time()
            has_summary, best_summary, sample_predictions, sample_counts, structure_info, adj_mat = self.get_decoded_outputs(batch, counter)
            #print('Time taken for decoder: ', time.process_time() - start)
            # token_contsel_tot_correct += token_consel_num_correct
            # token_contsel_tot_num += token_consel_num
            # sent_heads_tot_correct += sent_heads_num_correct
            # sent_heads_tot_num += sent_heads_num

            if args.predict_contsel_tags:
                no_words = batch.enc_word_lens[0]
                prediction = sample_predictions['token_contsel_prediction'][0:no_words]
                ref = batch.contsel_tags[0]
                write_tags(prediction, ref, counter, self._contsel_dir, self._contsel_ref_dir)
                counts['token_consel_num_correct'] += sample_counts['token_consel_num_correct']
                counts['token_consel_num'] += sample_counts['token_consel_num']

            if args.predict_sent_single_head:
                no_sents = batch.enc_doc_lens[0]
                prediction = sample_predictions['sent_single_heads_prediction'][0:no_sents].tolist()
                ref = batch.original_parent_heads[0]
                write_tags(prediction, ref, counter, self._sent_single_heads_dir, self._sent_single_heads_ref_dir)
                counts['sent_single_heads_num_correct'] += sample_counts['sent_single_heads_num_correct']
                counts['sent_single_heads_num'] += sample_counts['sent_single_heads_num']

            if args.predict_sent_all_head:
                counts['sent_all_heads_num_correct'] += sample_counts['sent_all_heads_num_correct']
                counts['sent_all_heads_num'] += sample_counts['sent_all_heads_num']

            if args.predict_sent_all_child:
                counts['sent_all_child_num_correct'] += sample_counts['sent_all_child_num_correct']
                counts['sent_all_child_num'] += sample_counts['sent_all_child_num']

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

            summary_len.append(len(decoded_words))
            assert adj_mat is not None, "Explicit matrix is none."
            assert structure_info['heads'] is not None, "Heads is none."
            precision, recall = tree_distance(structure_info['heads'], adj_mat.cpu().data.numpy()[0,:,:])
            if precision is not None and recall is not None:
                precision_tree_dist.append(precision)
                recall_tree_dist.append(recall)
            height_counter[structure_info['height']] += 1
            height_avg.append(structure_info['height'])
            leaf_node_proportion_avg.append(structure_info['leaf_nodes'])
            leaf_nodes_counter[np.floor(structure_info['leaf_nodes']*10)] += 1
            abstract_ref.append(" ".join(original_abstract_sents))
            abstract_pred.append(" ".join(decoded_words))

            sent_res = get_sent_dist(" ".join(decoded_words), batch.original_articles[0].decode(), minimum_seq=self.args.minimum_seq)

            sent_counter.append((sent_res['seen_sent'], sent_res['article_sent']))
            summary_len.append(sent_res['summary_len'])
            summary_sent.append(sent_res['summary_sent'])
            summary_sent_count[sent_res['summary_sent']] += 1
            article_sent.append(sent_res['article_sent'])
            if sent_res['avg_copied_seq_len'] is not None:
                avg_max_seq_len_list.append(sent_res['avg_copied_seq_len'])
                copied_sequence_per_sent.append(np.average(list(sent_res['counter_summary_sent_id'].values())))
            copied_sequence_len.update(sent_res['counter_copied_sequence_len'])
            sentence_copy_id_count.update(sent_res['counter_summary_sent_id'])
            article_copy_id_count_tot.update(sent_res['counter_article_sent_id'])
            novel_counter.update(sent_res['novel_ngram_counter'])
            repeated_counter.update(sent_res['repeated_ngram_counter'])

            sent_count_fp.write(str(counter)+"\t"+str(sent_res['article_sent'])+"\t"+str(sent_res['seen_sent'])+"\n")
            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)

            batch = self.batcher.next_batch()

            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()
            #print('Time taken for rest: ', time.process_time() - start)
            if args.decode_for_subset:
                if counter == 1000:
                    break

        print("Decoder has finished reading dataset for single_pass.")

        fp = open(self.stat_res_file, 'w')
        percentages = [float(len(seen_sent))/float(sent_count) for seen_sent, sent_count in sent_counter]
        avg_percentage = sum(percentages)/float(len(percentages))
        nosents = [len(seen_sent) for seen_sent, sent_count in sent_counter]
        avg_nosents = sum(nosents)/float(len(nosents))

        res = dict()
        res['avg_percentage_seen_sent'] = avg_percentage
        res['avg_nosents'] = avg_nosents
        res['summary_len'] = summary_sent_count
        res['avg_summary_len'] = np.average(summary_len)
        res['summary_sent'] = np.average(summary_sent)
        res['article_sent'] = np.average(article_sent)
        res['avg_copied_seq_len'] = np.average(avg_max_seq_len_list)
        res['avg_sequences_per_sent'] = np.average(copied_sequence_per_sent)
        res['counter_copied_sequence_len'] = copied_sequence_len
        res['counter_summary_sent_id'] = sentence_copy_id_count
        res['counter_article_sent_id'] = article_copy_id_count_tot
        res['novel_ngram_counter'] = novel_counter
        res['repeated_ngram_counter'] = repeated_counter

        fp.write("Summary metrics\n")
        for key in res:
            fp.write('{}: {}\n'.format(key, res[key]))

        fp.write("Structures metrics\n")
        fp.write("Average depth of RST tree: "+str(sum(height_avg)/len(height_avg))+"\n")
        fp.write("Average proportion of leaf nodes in RST tree: "+str(sum(leaf_node_proportion_avg)/len(leaf_node_proportion_avg))+"\n")
        fp.write("Precision of edges latent to explicit: "+str(np.average(precision_tree_dist))+"\n")
        fp.write("Recall of edges latent to explicit: "+str(np.average(recall_tree_dist))+"\n")
        fp.write("Tree height counter:\n")
        fp.write(str(height_counter) + "\n")
        fp.write("Tree leaf proportion counter:")
        fp.write(str(leaf_nodes_counter) + "\n")

        if args.predict_contsel_tags:
            fp.write("Avg token_contsel: "+str((counts['token_consel_num_correct']/float(counts['token_consel_num']))))
        if args.predict_sent_single_head:
            fp.write("Avg single sent heads: "+str((counts['sent_single_heads_num_correct']/float(counts['sent_single_heads_num']))))
        if args.predict_sent_all_head:
            fp.write("Avg all sent heads: "+str((counts['sent_all_heads_num_correct']/float(counts['sent_all_heads_num']))))
        if args.predict_sent_all_child:
            fp.write("Avg all sent child: "+str((counts['sent_all_child_num_correct']/float(counts['sent_all_child_num']))))
        fp.close()
        sent_count_fp.close()

        write_to_json_file(abstract_ref, self._rouge_ref_file)
        write_to_json_file(abstract_pred, self._rouge_pred_file)

    def get_decoded_outputs(self, batch, count):
        #batch should have only one example
        enc_batch, enc_padding_token_mask, enc_padding_sent_mask, enc_doc_lens, enc_sent_lens, \
            enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0, word_batch, word_padding_mask, enc_word_lens, \
                enc_tags_batch, enc_sent_tags, enc_sent_token_mat, adj_mat, weighted_adj_mat, norm_adj_mat, \
                    parent_heads, undir_weighted_adj_mat = get_input_from_batch(batch, use_cuda, self.args)


        enc_adj_mat = adj_mat
        if args.use_weighted_annotations:
            if args.use_undirected_weighted_graphs:
                enc_adj_mat = undir_weighted_adj_mat
            else:
                enc_adj_mat = weighted_adj_mat
         
        encoder_output = self.model.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask,
                                                         enc_padding_sent_mask, word_batch, word_padding_mask,
                                                         enc_word_lens, enc_tags_batch, enc_sent_token_mat,
                                                         enc_adj_mat)
        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, \
        enc_batch_extend_vocab, token_level_sentence_scores, sent_outputs, token_scores, \
        sent_scores, sent_matrix, sent_level_rep = \
                                    self.model.get_app_outputs(encoder_output, enc_padding_token_mask,
                                                   enc_padding_sent_mask, enc_batch_extend_vocab, enc_sent_token_mat)

        mask = enc_padding_sent_mask[0].unsqueeze(0).repeat(enc_padding_sent_mask.size(1),1) * enc_padding_sent_mask[0].unsqueeze(1).transpose(1,0)

        mask = torch.cat((enc_padding_sent_mask[0].unsqueeze(1), mask), dim=1)
        mat = encoder_output['sent_attention_matrix'][0][:,:] * mask

        structure_info = self.extract_structures(batch, encoder_output['token_attention_matrix'], mat, count, use_cuda, encoder_output['sent_score'])

        counts = {}
        predictions = {}
        if args.predict_contsel_tags:
            pred = encoder_output['token_score'][0, :, :].view(-1, 2)
            token_contsel_gold = enc_tags_batch[0, :].view(-1)
            token_contsel_prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
            token_contsel_prediction[token_contsel_gold==-1] = -2 # Explicitly set masked tokens as different from value in gold
            token_consel_num_correct = torch.sum(token_contsel_prediction.eq(token_contsel_gold)).item()
            token_consel_num = torch.sum(token_contsel_gold != -1).item()
            predictions['token_contsel_prediction'] = token_contsel_prediction
            counts['token_consel_num_correct'] = token_consel_num_correct
            counts['token_consel_num'] = token_consel_num

        if args.predict_sent_single_head:
            pred = encoder_output['sent_single_head_scores'][0, :, :]
            head_labels = parent_heads[0, :].view(-1)
            sent_single_heads_prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
            sent_single_heads_prediction[head_labels==-1] = -2 # Explicitly set masked tokens as different from value in gold
            sent_single_heads_num_correct = torch.sum(sent_single_heads_prediction.eq(head_labels)).item()
            sent_single_heads_num = torch.sum(head_labels != -1).item()
            predictions['sent_single_heads_prediction'] = sent_single_heads_prediction
            counts['sent_single_heads_num_correct'] = sent_single_heads_num_correct
            counts['sent_single_heads_num'] = sent_single_heads_num

        if args.predict_sent_all_head:
            pred = encoder_output['sent_all_head_scores'][0, :, :, :]
            target = adj_mat[0, :, :].permute(0,1).view(-1)
            sent_all_heads_prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
            sent_all_heads_prediction[target==-1] = -2 # Explicitly set masked tokens as different from value in gold
            sent_all_heads_num_correct = torch.sum(sent_all_heads_prediction.eq(target)).item()
            sent_all_heads_num = torch.sum(target != -1).item()
            predictions['sent_all_heads_prediction'] = sent_all_heads_prediction
            counts['sent_all_heads_num_correct'] = sent_all_heads_num_correct
            counts['sent_all_heads_num'] = sent_all_heads_num

        if args.predict_sent_all_child:
            pred = encoder_output['sent_all_child_scores'][0, :, :, :]
            target = adj_mat[0, :, :].view(-1)
            sent_all_child_prediction = torch.argmax(pred.clone().detach().requires_grad_(False), dim=1)
            sent_all_child_prediction[target==-1] = -2 # Explicitly set masked tokens as different from value in gold
            sent_all_child_num_correct = torch.sum(sent_all_child_prediction.eq(target)).item()
            sent_all_child_num = torch.sum(target != -1).item()
            predictions['sent_all_child_prediction'] = sent_all_child_prediction
            counts['sent_all_child_num_correct'] = sent_all_child_num_correct
            counts['sent_all_child_num'] = sent_all_child_num



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


            all_child, all_head = None, None
            if args.use_gold_annotations_for_decode:
                if args.use_weighted_annotations:
                    if args.use_undirected_weighted_graphs:
                        permuted_all_head = undir_weighted_adj_mat[:, :, :].permute(0,2,1)
                        all_head = permuted_all_head.clone()
                        row_sums = torch.sum(permuted_all_head, dim=2, keepdim=True)
                        all_head[row_sums.expand_as(permuted_all_head)!=0] = permuted_all_head[row_sums.expand_as(permuted_all_head)!=0]/row_sums.expand_as(permuted_all_head)[row_sums.expand_as(permuted_all_head)!=0]

                        base_all_child = undir_weighted_adj_mat[:, :, :]
                        all_child = base_all_child.clone()
                        row_sums = torch.sum(base_all_child, dim=2, keepdim=True)
                        all_child[row_sums.expand_as(base_all_child)!=0] = base_all_child[row_sums.expand_as(base_all_child)!=0]/row_sums.expand_as(base_all_child)[row_sums.expand_as(base_all_child)!=0]
                    else:
                        permuted_all_head = weighted_adj_mat[:, :, :].permute(0,2,1)
                        all_head = permuted_all_head.clone()
                        row_sums = torch.sum(permuted_all_head, dim=2, keepdim=True)
                        all_head[row_sums.expand_as(permuted_all_head)!=0] = permuted_all_head[row_sums.expand_as(permuted_all_head)!=0]/row_sums.expand_as(permuted_all_head)[row_sums.expand_as(permuted_all_head)!=0]

                        base_all_child = weighted_adj_mat[:, :, :]
                        all_child = base_all_child.clone()
                        row_sums = torch.sum(base_all_child, dim=2, keepdim=True)
                        all_child[row_sums.expand_as(base_all_child)!=0] = base_all_child[row_sums.expand_as(base_all_child)!=0]/row_sums.expand_as(base_all_child)[row_sums.expand_as(base_all_child)!=0]
                else:
                    permuted_all_head = adj_mat[:, :, :].permute(0,2,1)
                    all_head = permuted_all_head.clone()
                    row_sums = torch.sum(permuted_all_head, dim=2, keepdim=True)
                    all_head[row_sums.expand_as(permuted_all_head)!=0] = permuted_all_head[row_sums.expand_as(permuted_all_head)!=0]/row_sums.expand_as(permuted_all_head)[row_sums.expand_as(permuted_all_head)!=0]

                    base_all_child = adj_mat[:, :, :]
                    all_child = base_all_child.clone()
                    row_sums = torch.sum(base_all_child, dim=2, keepdim=True)
                    all_child[row_sums.expand_as(base_all_child)!=0] = base_all_child[row_sums.expand_as(base_all_child)!=0]/row_sums.expand_as(base_all_child)[row_sums.expand_as(base_all_child)!=0]
                    # all_head = adj_mat[:, :, :].permute(0,2,1) + 0.00005
                    # row_sums = torch.sum(all_head, dim=2, keepdim=True)
                    # all_head = all_head / row_sums
                    # all_child = adj_mat[:, :, :] + 0.00005
                    # row_sums = torch.sum(all_child, dim=2, keepdim=True)
                    # all_child = all_child / row_sums

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
                          coverage=(coverage_t_0[0] if self.args.is_coverage or self.args.bu_coverage_penalty else None))
                     for _ in range(args.beam_size)]

            while steps < args.max_dec_steps and len(results) < args.beam_size:
                latest_tokens = [h.latest_token for h in beams]
                # cur_len = torch.stack([len(h.tokens) for h in beams])
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
                if self.args.is_coverage or self.args.bu_coverage_penalty:
                    all_coverage = []
                    for h in beams:
                        all_coverage.append(h.coverage)
                    coverage_t_1 = torch.stack(all_coverage, 0)

                final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                                                                                        encoder_outputs, word_padding_mask, c_t_1,
                                                                                        extra_zeros, enc_batch_extend_vocab, coverage_t_1,
                                                                                        token_scores, sent_scores, sent_outputs,
                                                                                        enc_sent_token_mat, all_head,
                                                                                        all_child, sent_level_rep)

                if args.bu_coverage_penalty:
                    penalty = torch.max(coverage_t, coverage_t.clone().fill_(1.0)).sum(-1)
                    penalty -= coverage_t.size(-1)
                    final_dist -= args.beta*penalty.unsqueeze(1).expand_as(final_dist)
                if args.bu_length_penalty:
                    penalty = ((5 + steps+1) / 6.0) ** args.alpha
                    final_dist = final_dist/penalty


                topk_log_probs, topk_ids = torch.topk(final_dist, args.beam_size * 2)

                dec_h, dec_c = s_t
                dec_h = dec_h.squeeze()
                dec_c = dec_c.squeeze()

                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[i], dec_c[i])
                    context_i = c_t[i]
                    coverage_i = (coverage_t[i] if self.args.is_coverage or self.args.bu_coverage_penalty else None)

                    for j in range(args.beam_size * 2):  # for each of the top 2*beam_size hyps:
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
                    if len(beams) == args.beam_size or len(results) == args.beam_size:
                        break

                steps += 1

            if len(results) == 0:
                results = beams

            beams_sorted = self.sort_beams(results)

        return has_summary, beams_sorted[0], predictions, counts, structure_info, undir_weighted_adj_mat

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Structured Summarization Model')
    parser.add_argument('--save_path', type=str, default=None, help='location of the save path')
    parser.add_argument('--reload_path', type=str, default=None, help='location of the older saved path')
    parser.add_argument('--decode_data_path', type=str, default='/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/test.bin', help='location of the decode data path')
    parser.add_argument('--vocab_path', type=str, default=None, help='location of the eval data path')
    parser.add_argument('--use_small_train_data', action='store_true', default=False, help='use_small_data for training')

    parser.add_argument('--pointer_gen', action='store_true', default=False, help='use pointer-generator')
    parser.add_argument('--is_coverage', action='store_true', default=False, help='use coverage loss')
    parser.add_argument('--autoencode', action='store_true', default=False, help='use autoencoder setting')
    parser.add_argument('--reload_pretrained_clf_path', type=str, default=None, help='location of the older saved path')

    parser.add_argument('--sep_sent_features', action='store_true', default=False, help='use sent features for decoding attention')
    parser.add_argument('--token_scores', action='store_true', default=False, help='use token scores for decoding attention')
    parser.add_argument('--sent_scores', action='store_true', default=False, help='use sent scores for decoding attention')
    parser.add_argument('--fixed_scorer', action='store_true', default=False, help='use fixed pretrained scorer')
    parser.add_argument('--heuristic_chains', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--sm_ner_model', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_ner', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_coref', action='store_true', default=False, help='heuristic coref for training')
    parser.add_argument('--max_dec_steps', type=int, default=100, help='Max Dec Steps')
    parser.add_argument('--beam_size', type=int, default=3, help='Max Dec Steps')
    parser.add_argument('--use_glove', action='store_true', default=False, help='use_glove_embeddings for training')

    parser.add_argument('--predict_summaries', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_sent_single_head', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_sent_all_head', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_sent_all_child', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--predict_contsel_tags', action='store_true', default=False, help='decode summarization')

    parser.add_argument('--sent_attention_at_dec', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_all_sent_head_at_decode', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_all_sent_child_at_decode', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_single_sent_head_at_decode', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_gold_annotations_for_decode', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_weighted_annotations', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--use_undirected_weighted_graphs', action='store_true', default=False, help='decode summarization')

    parser.add_argument('--use_coref_param', action='store_true', default=False, help='decode summarization')
    parser.add_argument('--no_latent_str', action='store_true', default=False, help='decode summarization')

    parser.add_argument('--use_sent_single_head_loss', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_sent_all_head_loss', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--use_sent_all_child_loss', action='store_true', default=False, help='heuristic ner for training')

    parser.add_argument('--decode_for_subset', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--bu_coverage_penalty', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--bu_length_penalty', action='store_true', default=False, help='heuristic ner for training')
    parser.add_argument('--beta', type=int, default=5, help='heuristic ner for training')
    parser.add_argument('--alpha', type=float, default=0.8, help='heuristic ner for training')

    parser.add_argument('--use_coref_att_encoder', action='store_true', default=False, help='decode summarization')

    args = parser.parse_args()
    model_filename = args.reload_path
    save_path = args.save_path
    beam_Search_processor = BeamSearch(args, model_filename, save_path)
    beam_Search_processor.decode()