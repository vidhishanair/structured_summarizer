# Content of this file is copied from https://github.com/atulkum/pointer_summarizer/blob/master/

import queue
import time
from random import shuffle
from threading import Thread
import itertools

import numpy as np
# import tensorflow as tf

import utils.config as config
import utils.data as data

import random

random.seed(1234)


class Example(object):

    def __init__(self, article, abstract_sentences, tags, vocab, args):
        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)
        self.pointer_gen = args.pointer_gen

        # Process the article
        # article_words = article.split()
        if args.autoencode == True:
            article_sents = abstract_sentences
            article_sents = article_sents[:10]
            article_words = [sent.split()[:20] for sent in article_sents]
            all_article_words = ' '.join(abstract_sentences).split()
            article_word_tags = []
        else:
            article_sents_tmp = article.decode().split('<split1>')
            sent_tags = tags.decode().split('<split1>')
            size = 0
            article_sents = []
            article_sent_tags = []

            # for sent, tags in zip(article_sents_tmp, sent_tags):
            #     sent = sent.split()
            #     tags = tags.split()
            #     if len(sent) + size <= config.max_enc_steps:
            #         article_sents.append(sent)
            #         article_sent_tags.append([int(x) for x in tags])
            #         size += len(sent)
            #     elif size >= config.max_enc_steps:
            #         break
            #     else:
            #         article_sents.append(sent[:config.max_enc_steps-size])
            #         article_sent_tags.append([int(x) for x in tags[:config.max_enc_steps-size]])

            article_sents = article_sents_tmp[:20]
            article_sent_tags = sent_tags[:20]
            article_words = [sent.split()[:140] for sent in article_sents]
            article_word_tags = [[int(x) for x in sent.split()[:140]] for sent in article_sent_tags]
            all_article_words = list(itertools.chain.from_iterable(article_words))

            #article_sents = article_sents
            #article_words = [sent.split() for sent in article_sents]
        # if len(article_words) > config.max_enc_steps:
        #   article_words = article_words[:config.max_enc_steps]
        self.enc_tok_len = [len(sent) for sent in article_words]  # store the length after truncation but before padding
        self.enc_doc_len = len(article_words)
        self.enc_word_len = len(all_article_words)
        # self.enc_input = [vocab.word2id(w) for sent in article_words for w in sent] # list of word ids; OOVs are represented by the id for UNK token
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
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if args.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            if args.test_sent_matrix:
                self.enc_input_extend_vocab, self.article_oovs = data.article2ids(all_article_words, vocab)
            else:
                self.enc_input_extend_vocab, self.article_oovs = data.sent_sep_article2ids(article_words, vocab)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Store the original strings
        self.enc_tags = article_word_tags
        self.original_article = article
        self.article_words = article_words
        self.original_abstract = abstract
        self.original_abstract_sents = abstract_sentences

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
                self.enc_tags[i].append(-1)
        # if self.pointer_gen:
        #     for i in range(0, len(self.enc_input_extend_vocab)):
        #         while len(self.enc_input_extend_vocab[i]) < max_len:
        #             self.enc_input_extend_vocab[i].append(pad_id)

    def pad_encoder_docs(self, max_len, pad_id, max_tok_len):
        while len(self.enc_input) < max_len:
            self.enc_input.append([pad_id] * max_tok_len)
            self.enc_tags.append([-1] * max_tok_len)
        # if self.pointer_gen:
        #     while len(self.enc_input_extend_vocab) < max_len:
        #         self.enc_input_extend_vocab.append([pad_id] * max_tok_len)

    def pad_encoder_words(self, max_len, pad_id):
        while len(self.word_input) < max_len:
            self.word_input.append(pad_id)
        if self.pointer_gen:
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
    def __init__(self, example_list, vocab, batch_size, args):
        self.batch_size = batch_size
        self.pointer_gen = args.pointer_gen
        self.test_sent_matrix = args.test_sent_matrix
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        self.store_orig_strings(example_list)  # store the original strings

    def init_encoder_seq(self, example_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_tok_len = max([max(ex.enc_tok_len) for ex in example_list])
        max_enc_doc_len = max([ex.enc_doc_len for ex in example_list])
        max_enc_word_len = max([ex.enc_word_len for ex in example_list])
        #total_tok_len = sum([ex.enc_word_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_encoder_tokens(max_enc_tok_len, self.pad_id)
        for ex in example_list:
            ex.pad_encoder_words(max_enc_word_len, self.pad_id)
        for ex in example_list:
            ex.pad_encoder_docs(max_enc_doc_len, self.pad_id, max_enc_tok_len)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        # self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)

        self.enc_batch = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.int32)
        self.enc_tags_batch = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.int32)

        self.enc_word_batch = np.zeros((self.batch_size, max_enc_word_len), dtype=np.int32)
        self.enc_word_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_doc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_sent_lens = np.ones((self.batch_size, max_enc_doc_len), dtype=np.int32)
        self.enc_sent_token_marker = np.zeros((self.batch_size, max_enc_doc_len, max_enc_word_len), dtype=np.float32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.float32)
        self.enc_padding_token_mask = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.float32)
        self.enc_padding_sent_mask = np.zeros((self.batch_size, max_enc_doc_len), dtype=np.float32)
        self.enc_padding_word_mask = np.zeros((self.batch_size, max_enc_word_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.enc_batch[i, :] = np.array(ex.enc_input)
            self.enc_tags_batch[i, :] = np.array(ex.enc_tags)
            self.enc_word_batch[i,:] = np.array(ex.word_input)
            self.enc_doc_lens[i] = ex.enc_doc_len
            self.enc_word_lens[i] = ex.enc_word_len
            word_counter = 0
            for j in range(len(ex.enc_tok_len)):
                self.enc_sent_lens[i][j] = ex.enc_tok_len[j]
                for k in range(ex.enc_tok_len[j]):
                    self.enc_padding_mask[i][j][k] = 1
                    self.enc_padding_token_mask[i][j][k] = 1
                    self.enc_sent_token_marker[i][j][word_counter] = 1
                    word_counter+=1
                self.enc_padding_sent_mask[i][j] = 1
            for j in range(ex.enc_word_len):
                self.enc_padding_word_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if self.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            if self.test_sent_matrix:
                self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_word_len), dtype=np.int32)
            else:
                self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_doc_len, max_enc_tok_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.enc_batch_extend_vocab[i, :] = np.array(ex.enc_input_extend_vocab[:])

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
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
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists


class Batcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, mode, batch_size, single_pass, args):
        self._data_path = data_path
        self._vocab = vocab
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
            self._num_example_q_threads = 1  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 1  # 4  # num threads to fill batch queue
            self._bucketing_cache_size = 1  # 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
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
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

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
        input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

        while True:
            try:
                (article, abstract, tags) = next(input_gen)  # read the next example from file. article and abstract are both strings.
            except StopIteration:  # if there are no more examples:
                print("The example generator for this example queue filling thread has exhausted data.")
                if self._single_pass:
                    print("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
                    self._finished_reading = True
                    break
                else:
                    raise Exception("single_pass mode is off but the example generator is out of data; error.")

            abstract_sentences = [sent.strip() for sent in data.abstract2sents(
                abstract)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
            example = Example(article, abstract_sentences, tags, self._vocab, self.args)  # Process into an Example.
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
                    self._batch_queue.put(Batch(b, self._vocab, self.batch_size, self.args))

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

    def text_generator(self, example_generator):
        while True:
            e = next(example_generator)  # e is a tf.Example
            try:
                article_text = e.features.feature['article'].bytes_list.value[
                    0]  # the article text was saved under the key 'article' in the data files
                tags = e.features.feature['labels'].bytes_list.value[
                    0]  # the article text was saved under the key 'article' in the data files
                abstract_text = e.features.feature['abstract'].bytes_list.value[
                    0]  # the abstract text was saved under the key 'abstract' in the data files
            except:# ValueError:
                # tf.logging.error('Failed to get article or abstract from example')
                print(article_text)
                print(e.features.feature['labels'].bytes_list.value)
                exit()
                print('Failed to get article or abstract from example')
                continue
            if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                #print('Found an example with empty article text. Skipping it.')
                continue
            else:
                #print(article_text)
                #print(e.features.feature['labels'].bytes_list.value)
                #exit()
                #tags = e.features.feature['labels'].bytes_list.value[
                #    0]
                yield (article_text, abstract_text, tags)
