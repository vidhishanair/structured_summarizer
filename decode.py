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
from utils.utils import write_for_rouge, rouge_eval, rouge_log, write_to_json_file
from utils.train_util import get_input_from_batch, get_output_from_batch
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.coco import COCO


use_cuda = config.use_gpu and torch.cuda.is_available()

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
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        self._rouge_ref_file = os.path.join(self._decode_dir, 'rouge_ref.json')
        self._rouge_pred_file = os.path.join(self._decode_dir, 'rouge_pred.json')
        for p in [self._decode_dir, self._structures_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True, args=args)
        time.sleep(15)

        self.model = Model(args)
        self.model.eval()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def extract_structures(self, batch, sent_attention_matrix, doc_attention_matrix, count, use_cuda):
        fileName = os.path.join(self._structures_dir, str(count)+".txt")
        fp = open(fileName, "w")
        fp.write("Doc: "+str(count)+"\n")
        #exit(0)
        doc_attention_matrix = doc_attention_matrix[:,:,1:] #this change yet to be tested!
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

        shape2 = doc_attention_matrix[0][0:l,0:l].size()
        row = torch.ones([1, shape2[1]+1]).cuda()
        column = torch.zeros([shape2[0], 1]).cuda()
        scores = doc_attention_matrix[0][0:l, 0:l]
        new_scores = torch.cat([column, scores], dim=1)
        new_scores = torch.cat([row, new_scores], dim=0)
        heads, tree_score = chu_liu_edmonds(new_scores.data.cpu().numpy().astype(np.float64))
        #print(heads, tree_score)
        fp.write("\n")
        fp.write(str(heads)+" ")
        fp.write(str(tree_score)+"\n")
        fp.close()

    def decode(self):
        start = time.time()
        counter = 0
        abstract_ref = []
        abstract_pred = []
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            boo, best_summary = self.beam_search(batch, counter)
            if boo==False:
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

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting PYCOCO - ROUGE eval...")
        #results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        #rouge_log(results_dict, self._decode_dir)
        write_to_json_file(abstract_ref, self._rouge_ref_file)
        write_to_json_file(abstract_pred, self._rouge_pred_file)
        cocoEval = COCOEvalCap(self._rouge_ref_file, self._rouge_pred_file)
        cocoEval.evaluate()
        for metric, score in cocoEval.eval.items():
            print('%s: %.3f'%(metric, score))


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
        enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, word_batch, word_padding_mask, enc_word_lens \
            = get_input_from_batch(batch, use_cuda, args)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_output = self.model.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask,
                                                         enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens)
        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_prediction, sent_outputs = \
            self.get_app_outputs(encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab)


    def beam_search(self, batch, count):
        #batch should have only one example
        enc_batch, enc_padding_token_mask, enc_padding_sent_mask,  enc_doc_lens, enc_sent_lens, enc_batch_extend_vocab, \
        extra_zeros, c_t_0, coverage_t_0, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch = \
            get_input_from_batch(batch, use_cuda, self.args)

        if(enc_batch.size()[1]==1 or enc_batch.size()[2]==1):
            return False, None

        encoder_output = self.model.encoder.forward_test(enc_batch,enc_sent_lens,enc_doc_lens,enc_padding_token_mask,
                                                         enc_padding_sent_mask, word_batch, word_padding_mask, enc_word_lens, enc_tags_batch)
        encoder_outputs, enc_padding_mask, encoder_last_hidden, max_encoder_output, enc_batch_extend_vocab, token_level_sentence_scores, sent_prediction, sent_outputs = \
            self.get_app_outputs(encoder_output, enc_padding_token_mask, enc_padding_sent_mask, enc_batch_extend_vocab)

        self.extract_structures(batch, encoder_output['token_attention_matrix'], encoder_output['sent_attention_matrix'], count, use_cuda)
        print(encoder_output['sent_importance_vector'])

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
        results = []
        steps = 0
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
                                                                                    encoder_outputs, enc_padding_mask, c_t_1,
                                                                                    extra_zeros, enc_batch_extend_vocab, coverage_t_1, token_level_sentence_scores, sent_outputs)

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

        return True, beams_sorted[0]

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
    parser.add_argument('--gold_tag_scores', action='store_true', default=False, help='use gold tags for scores')
    parser.add_argument('--decode_setting', action='store_true', default=False, help='use gold tags for scores')
    parser.add_argument('--sep_sent_features', action='store_true', default=False, help='use sent features for decoding attention')

    args = parser.parse_args()
    model_filename = args.reload_path
    save_path = args.save_path
    beam_Search_processor = BeamSearch(args, model_filename, save_path)
    beam_Search_processor.decode()


