# Content of this file is copied from https://github.com/atulkum/pointer_summarizer/blob/master/

import queue
import time
from random import shuffle
from threading import Thread
import itertools

import numpy as np
import utils.config as config
import utils.data as data

import ast

import random

random.seed(config.seed)


class Example(object):

    def __init__(self, article, abstract_sentences, tags, links, vocab, args):
        self.args = args
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        self.pointer_gen = args.pointer_gen

        # Process the article
        words = article.decode().split()[:700]
        tags = tags.decode().split()[:700]
        sent = [[]]
        sent_tags = [[]]
        for word, tag in list(zip(words, tags)):
            if word == "<split1>":
                sent.append([])
                sent_tags.append([])
            else:
                sent[-1].append(word)
                sent_tags[-1].append(tag)
        article_words = sent[:20]
        #for idx, sent in enumerate(article_words):
        #    print(str(idx)+"  "+" ".join(sent))
        article_word_tags = sent_tags[:20]
        all_article_words = list(itertools.chain.from_iterable(article_words))
        all_article_tags = list(itertools.chain.from_iterable(article_word_tags))

        #article_sents_tmp = article.decode().split('<split1>')
        #sent_tags = tags.decode().split('<split1>')
        #article_sents = article_sents_tmp[:20]
        #article_sent_tags = sent_tags[:20]
        #article_words = [sent.split()[:140] for sent in article_sents]
        #article_word_tags = [[int(x) for x in sent.split()[:140]] for sent in article_sent_tags]
        #all_article_words = list(itertools.chain.from_iterable(article_words)) 

        self.enc_tok_len = [len(sent) for sent in article_words]  # store the length after truncation but before padding
        self.enc_doc_len = len(article_words)
        self.enc_word_len = len(all_article_words)

        # list of word ids; OOVs are represented by the id for UNK token
        self.enc_input = []
        for sent in article_words:
            self.enc_input.append([vocab.word2id(w) for w in sent])
        self.word_input = []
        for w in all_article_words:
            self.word_input.append(vocab.word2id(w))

        # Process the abstract
        abstract = ' '.join(abstract_sentences)  # string
        abstract_words = abstract.split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in
                   abstract_words]  # list of word ids; OOVs are represented by the id for UNK token


        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, self.args.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if args.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.enc_input_extend_vocab, self.article_oovs = data.article2ids(all_article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, self.args.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Create adj_mat for supervision
        if self.args.heuristic_chains:
            # self.sup_adj_mat, self.parent_heads = self.generate_adj_mat_sup(len(article_words), links)
            self.adj_mat, self.weighted_adj_mat, self.norm_adj_mat, \
                self.parent_heads, self.undir_weighted_adj_mat = self.generate_adj_mat_sup(len(article_words), links)

        # Store the original strings
        self.enc_tags = all_article_tags
        self.enc_sent_tags = article_word_tags
        self.original_article = article
        self.article_words = article_words
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences


    # def generate_adj_mat_sup(self, no_sents, links):
    #     adj_mat = np.zeros((no_sents, no_sents), dtype='float32')
    #     parent_heads = np.full(no_sents, fill_value=-1, dtype='int')
    #     for link in links:
    #         if self.args.link_id_typed:
    #             type = link[3]
    #             if type == 'ner':
    #                 weight = 1
    #             else:
    #                 weight = 0.5
    #         else:
    #             weight = 1
    #         parent = link[1]
    #         child = link[0]
    #         if parent >= no_sents or child >= no_sents:
    #             continue
    #         adj_mat[parent][child] += weight
    #
    #     adjusted_adj_mat = adj_mat + config.eps
    #     row_sums = adjusted_adj_mat.sum(axis=0)
    #     norm_adj_mat = adjusted_adj_mat / row_sums[np.newaxis, :] # eq prob on all incase of no head is bad.
    #
    #     for sent_idx in range(no_sents):
    #         head_dist = adj_mat[:, sent_idx]
    #         max_score = np.max(head_dist)
    #         if max_score <= config.eps:
    #             continue
    #         else:
    #             indices = np.asarray(head_dist==max_score).nonzero()[0]
    #             head = indices[(np.abs(indices - sent_idx)).argmin()]
    #         parent_heads[sent_idx] = head
    #     return norm_adj_mat, parent_heads

    def generate_adj_mat_sup(self, no_sents, links):
        adj_mat = np.zeros((no_sents, no_sents), dtype='float32')
        weighted_adj_mat = np.zeros((no_sents, no_sents), dtype='float32')
        undir_weighted_adj_mat = np.zeros((no_sents, no_sents), dtype='float32')
        parent_heads = np.full(no_sents, fill_value=-1, dtype='int')

        if self.args.sm_ner_model:
            for link in links:
                parent = link[1]
                child = link[0]
                if parent >= no_sents or child >= no_sents:
                    continue
                adj_mat[parent][child] = 1
                weighted_adj_mat[parent][child] += 1
                undir_weighted_adj_mat[parent][child] += 1
                undir_weighted_adj_mat[child][parent] += 1

            adjusted_adj_mat = weighted_adj_mat + config.eps
            row_sums = adjusted_adj_mat.sum(axis=0)
            norm_adj_mat = adjusted_adj_mat / row_sums[np.newaxis, :] # eq prob on all incase of no head is bad.

            for sent_idx in range(no_sents):
                head_dist = weighted_adj_mat[:, sent_idx]
                max_score = np.max(head_dist)
                if max_score <= config.eps:
                    continue
                else:
                    indices = np.asarray(head_dist == max_score).nonzero()[0]
                    head = indices[(np.abs(indices - sent_idx)).argmin()]
                parent_heads[sent_idx] = head
        else:
            if self.args.use_ner:
                for link in links['ner']:
                    parent = link['head_id']
                    child = link['tail_id']
                    if parent >= no_sents or child >= no_sents:
                        continue
                    adj_mat[parent][child] = 1
                    weighted_adj_mat[parent][child] += 1
                    undir_weighted_adj_mat[parent][child] += 1
                    undir_weighted_adj_mat[child][parent] += 1

            if self.args.use_coref:
                for link in links['coref']:
                    parent = link['head_id']
                    child = link['tail_id']
                    if parent >= no_sents or child >= no_sents:
                        continue
                    #print(parent, child)
                    adj_mat[parent][child] = 1
                    weighted_adj_mat[parent][child] += 1
                    undir_weighted_adj_mat[parent][child] += 1
                    undir_weighted_adj_mat[child][parent] += 1
            #print(weighted_adj_mat)
            adjusted_adj_mat = weighted_adj_mat + config.eps
            row_sums = adjusted_adj_mat.sum(axis=0)
            norm_adj_mat = adjusted_adj_mat / row_sums[np.newaxis, :] # eq prob on all incase of no head is bad.

            for sent_idx in range(no_sents):
                head_dist = weighted_adj_mat[:, sent_idx]
                max_score = np.max(head_dist)
                if max_score <= config.eps:
                    continue
                else:
                    indices = np.asarray(head_dist == max_score).nonzero()[0]
                    head = indices[(np.abs(indices - sent_idx)).argmin()]
                parent_heads[sent_idx] = head
        #print("here: ", links, adj_mat)
        #exit()
        return adj_mat, weighted_adj_mat, norm_adj_mat, parent_heads, undir_weighted_adj_mat

    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_tokens(self, max_len, pad_id):
        for i in range(0, len(self.enc_input)):
            while len(self.enc_input[i]) < max_len:
                self.enc_input[i].append(pad_id)
                self.enc_sent_tags[i].append(-1)
        # if self.pointer_gen:
        #     for i in range(0, len(self.enc_input_extend_vocab)):
        #         while len(self.enc_input_extend_vocab[i]) < max_len:
        #             self.enc_input_extend_vocab[i].append(pad_id)

    def pad_encoder_docs(self, max_len, pad_id, max_tok_len):
        while len(self.enc_input) < max_len:
            self.enc_input.append([pad_id] * max_tok_len)
            self.enc_sent_tags.append([-1] * max_tok_len)
        # if self.args.heuristic_chains:
        #     self.sup_adj_mat.append([0]*)

        # if self.pointer_gen:
        #     while len(self.enc_input_extend_vocab) < max_len:
        #         self.enc_input_extend_vocab.append([pad_id] * max_tok_len)

    def pad_encoder_words(self, max_len, pad_id):
        while len(self.word_input) < max_len:
            self.word_input.append(pad_id)
            self.enc_tags.append(-1)
        if self.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size, args):
        self.args = args
        self.batch_size = batch_size
        self.pointer_gen = args.pointer_gen
        self.heuristic_chains = args.heuristic_chains
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_tok_len = max([max(ex.enc_tok_len) for ex in example_list])
        max_enc_doc_len = max([ex.enc_doc_len for ex in example_list])
        max_enc_word_len = max([ex.enc_word_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_tokens(max_enc_tok_len, self.pad_id)
        for ex in example_list:
            ex.pad_encoder_words(max_enc_word_len, self.pad_id)
        for ex in example_list:
            ex.pad_encoder_docs(max_enc_doc_len, self.pad_id, max_enc_tok_len)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.

        self.enc_batch = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.int32)
        self.enc_sent_tags = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.int32)
        self.enc_tags_batch = np.zeros((self.batch_size, max_enc_word_len), dtype=np.int32)
        self.enc_word_batch = np.zeros((self.batch_size, max_enc_word_len), dtype=np.int32)
        self.enc_word_lens = np.zeros(self.batch_size, dtype=np.int32)
        self.enc_doc_lens = np.zeros(self.batch_size, dtype=np.int32)
        self.enc_sent_lens = np.ones((self.batch_size, max_enc_doc_len), dtype=np.int32)
        self.enc_sent_token_marker = np.zeros((self.batch_size, max_enc_doc_len, max_enc_word_len), dtype=np.float32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.float32)
        self.enc_padding_token_mask = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.float32)
        self.enc_padding_sent_mask = np.zeros((self.batch_size, max_enc_doc_len), dtype=np.float32)
        self.enc_padding_word_mask = np.zeros((self.batch_size, max_enc_word_len), dtype=np.float32)

        if self.heuristic_chains:
            self.adj_mat = np.full((self.batch_size, max_enc_doc_len, max_enc_doc_len), fill_value=0, dtype=np.float32)
            self.weighted_adj_mat = np.full((self.batch_size, max_enc_doc_len, max_enc_doc_len), fill_value=0, dtype=np.float32)
            self.norm_adj_mat = np.zeros((self.batch_size, max_enc_doc_len, max_enc_doc_len), dtype=np.float32)
            self.parent_heads = np.full((self.batch_size, max_enc_doc_len), fill_value=0, dtype=np.int32)
            self.undir_weighted_adj_mat = np.full((self.batch_size, max_enc_doc_len, max_enc_doc_len), fill_value=0, dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = np.array(ex.enc_input)
            self.enc_sent_tags[i,:] = np.array(ex.enc_sent_tags)
            self.enc_tags_batch[i, :] = np.array(ex.enc_tags)
            self.enc_word_batch[i,:] = np.array(ex.word_input)
            self.enc_doc_lens[i] = ex.enc_doc_len
            self.enc_word_lens[i] = ex.enc_word_len
            word_counter = 0

            for j in range(len(ex.enc_tok_len)):
                # self.enc_sent_lens[i][j] = ex.enc_tok_len[j]
                for k in range(ex.enc_tok_len[j]):
                    self.enc_padding_mask[i][j][k] = 1
                    self.enc_padding_token_mask[i][j][k] = 1
                    self.enc_sent_token_marker[i][j][word_counter] = 1
                    word_counter+=1
                self.enc_padding_sent_mask[i][j] = 1
            for j in range(ex.enc_word_len):
                self.enc_padding_word_mask[i][j] = 1

            if self.heuristic_chains:
                self.adj_mat[i, :ex.adj_mat.shape[0], :ex.adj_mat.shape[1]] = ex.adj_mat
                self.weighted_adj_mat[i, :ex.weighted_adj_mat.shape[0], :ex.weighted_adj_mat.shape[1]] = ex.weighted_adj_mat
                self.undir_weighted_adj_mat[i, :ex.undir_weighted_adj_mat.shape[0], :ex.undir_weighted_adj_mat.shape[1]] = ex.undir_weighted_adj_mat
                self.norm_adj_mat[i, :ex.norm_adj_mat.shape[0], :ex.norm_adj_mat.shape[1]] = ex.norm_adj_mat
                self.parent_heads[i, :ex.parent_heads.shape[0]] = ex.parent_heads

        # For pointer-generator mode, need to store some extra info
        if self.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_word_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = np.array(ex.enc_input_extend_vocab[:])

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(self.args.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, self.args.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, self.args.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, self.args.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.orifinal_article_words = [ex.article_words for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists
        if self.args.heuristic_chains:
            self.original_parent_heads = [ex.parent_heads for ex in example_list]  # list of list of lists
        self.contsel_tags = [ex.enc_tags for ex in example_list]  # list of list of lists


class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, mode, batch_size, single_pass, args):
        self._data_path = data_path
        self._vocab = vocab
        self.heuristic_chains = args.heuristic_chains
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        self.args = args
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 16  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 4  # 4  # num threads to fill batch queue
            self._bucketing_cache_size = 100  # 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        if mode == 'train':
            self.setup_queues()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def setup_queues(self):
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        time.sleep(5)
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()
        time.sleep(10)
        #print('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
        #        self._batch_queue.qsize(), self._example_queue.qsize())

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            print('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                  self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                print("Finished reading dataset in single_pass mode.")
                return None
        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass, self.args))

        while True:
            try:
                (article, abstract, tags, links) = next(input_gen)  # read the next example from file. article and abstract are both strings.
            except Exception as ex:  # if there are no more examples: #In python 3.7 StopIteration is a RuntimeError
                print(ex)
                print("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    print("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in data.abstract2sents(
                abstract)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
            example = Example(article, abstract_sentences, tags, links, self._vocab, self.args)  # Process into an Example.
            self._example_queue.put(example)  # place the Example in the example queue.

    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(Batch(b, self._vocab, self.batch_size, self.args))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.enc_doc_len,
                                reverse=True)  # sort by length of encoder sequence

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    bat = Batch(b, self._vocab, self.batch_size, self.args)
                    self._batch_queue.put(bat)
                    #self.tot += bat.num_word_tags

    def watch_threads(self):
        while True:
            # print(
            #     'Bucket queue size: %i, Input queue size: %i',
            #     self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    print('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator, dataset='cnn-dm'):
        if dataset == 'cnn-dm':
            while True:
                e = next(example_generator)  # e is a tf.Example
                try:
                    article_text = e.features.feature['article'].bytes_list.value[
                        0]  # the article text was saved under the key 'article' in the data files
                    tags = e.features.feature['labels'].bytes_list.value[
                        0]  # the article text was saved under the key 'article' in the data files
                    abstract_text = e.features.feature['abstract'].bytes_list.value[
                        0]  # the abstract text was saved under the key 'abstract' in the data files
                    links = []
                    if self.heuristic_chains:
                        links = e.features.feature['links'].bytes_list.value[
                            0]  #
                        if links != b"":
                            links = ast.literal_eval(links.decode('utf-8'))
                        #print(links)
                        #print(links['coref'])
                        #print(len(links['ner']))
                except:# ValueError:
                    print(article_text)
                    print(e.features.feature['labels'].bytes_list.value)
                    exit()
                    print('Failed to get article or abstract from example')
                    continue
                if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                    #print('Found an example with empty article text. Skipping it.')
                    continue
                else:
                    yield (article_text, abstract_text, tags, links)
        else:
            print("Dataset value must be one of 'cnn-dm'/ 'inshorts' / 'gigaword' ")
