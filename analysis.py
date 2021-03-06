import os
from tqdm import tqdm
import spacy
from collections import Counter, defaultdict
import numpy as np
import argparse

nlp = spacy.load("en_core_web_lg")

def compile_substring(start, end, split):
    """
    Return the substring starting at start and ending at end.
    """
    if start == end:
        return split[start]
    return " ".join(split[start:end + 1])

def get_sent_dist(summary, article, minimum_seq=3):
    """
    Returns the number of sentences that were copied from the article,
    and the number of sentences in the article.

    Considers a match if there is a common sequence of at least 3 words.
    """
    article_list = [sent.split() for sent in article.split("<split1>")]
    ssplit = summary.split()
    startix = 0
    endix = 0
    matchstrings = Counter()
    seen_sentences = set()
    sentence_copy_id_count = Counter()
    copied_sequence_len = Counter()
    article_copy_id_count = Counter()

    sum_tok = nlp(summary)
    summary_sent_end = [sent.end for sent in sum_tok.sents]
    longest_match_list = []
    while endix < len(ssplit) + 1:
        match_found = False
        if endix < len(ssplit):
            searchstring = compile_substring(startix, endix, ssplit)
            current_match_list = []
            for idx, sent in enumerate(article_list):
                if searchstring in " ".join(sent):
                    match_found = True
                    current_match_list.append(idx)
        if match_found:
            endix += 1
            longest_match_list = current_match_list
        else:
            # only phrases of length minimum_seq+ words should be considered.
            if startix >= endix - (minimum_seq - 1):
                startix += 1
                endix = startix
            else:
                full_string = compile_substring(startix, endix - 1, ssplit)
                if matchstrings[full_string] >= 1:
                    pass
                else:
                    seen_sentences.update(longest_match_list)
                    matchstrings[full_string] += 1

                copied_sequence_len[endix - startix] += 1
                sentence_index = summary_sent_end.index(min(i for i in summary_sent_end if i > endix-1))
                # Count the number of mentions for sentences of a given index.
                sentence_copy_id_count[sentence_index] += 1

                # Update the count of each matched sentence in the article.
                for sent_id in longest_match_list:
                    article_copy_id_count[sent_id] += 1

                longest_match_list = []
                startix = endix

    avg_max_seq_len = None
    if len(copied_sequence_len.values()) != 0:
        # List of subsequence length times the number of occurrences of that subsequence.
        tot_seq_len = [key * copied_sequence_len[key] for key in copied_sequence_len.keys()]
        # Get the average by summing and deviding by the total number of subsequence matches.
        avg_max_seq_len = sum(tot_seq_len) / sum(copied_sequence_len.values())

    novel_counter, repeated_counter = get_novel_ngram_count(summary, sum_tok, article)

    res = dict()
    res['seen_sent'] = list(seen_sentences)
    res['summary_len'] = len(sum_tok)
    res['summary_sent'] = len(list(sum_tok.sents))
    res['article_sent'] = len(article_list)
    res['avg_copied_seq_len'] = avg_max_seq_len
    res['counter_copied_sequence_len'] = copied_sequence_len
    res['counter_summary_sent_id'] = sentence_copy_id_count
    res['counter_article_sent_id'] = article_copy_id_count
    res['novel_ngram_counter'] = novel_counter
    res['repeated_ngram_counter'] = repeated_counter
    return res


def get_novel_ngram_count(summary, sum_tok, article):
    repeated_counter = Counter()
    novel_counter = Counter()
    asplit = article.replace('<split1>', '').split()
    ssplit = summary.split()

    # 1grams
    repeated = set(ssplit).intersection(set(asplit))
    novel = set(ssplit) - repeated
    repeated_counter[1] = len(repeated)
    novel_counter[1] = len(novel)

    # 2-gram
    sum_2gram = list(zip(ssplit, ssplit[1:]))
    art_2gram = list(zip(asplit, asplit[1:]))
    repeated = set(sum_2gram).intersection(set(art_2gram))
    novel = set(sum_2gram) - repeated
    repeated_counter[2] = len(repeated)
    novel_counter[2] = len(novel)

    # 3-gram
    sum_3gram = list(zip(ssplit, ssplit[1:], ssplit[2:]))
    art_3gram = list(zip(asplit, asplit[1:], asplit[2:]))
    repeated = set(sum_3gram).intersection(set(art_3gram))
    novel = set(sum_3gram) - repeated
    repeated_counter[3] = len(repeated)
    novel_counter[3] = len(novel)

    # 4 gram
    sum_4gram = list(zip(ssplit, ssplit[1:], ssplit[2:], ssplit[3:]))
    art_4gram = list(zip(asplit, asplit[1:], asplit[2:], asplit[3:]))
    repeated = set(sum_4gram).intersection(set(art_4gram))
    novel = set(sum_4gram) - repeated
    repeated_counter[4] = len(repeated)
    novel_counter[4] = len(novel)

    # sentence
    for summary_sent in sum_tok.sents:
        if summary_sent.text.strip() in article:
            repeated_counter[-1] += 1
        else:
            novel_counter[-1] += 1

    return novel_counter, repeated_counter

def get_avg_sent_copied(article_dir, summary_dir, minimum_seq=2, file_suff='decoded'):
    article_files = os.listdir(article_dir)
    # summary_files = os.listdir(summary_dir)
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
    for article in tqdm(article_files):
        counter = article.split("_")[0]
        summ = str(counter)+"_{}.txt".format(file_suff)
        article = open(os.path.join(article_dir, article)).read()
        summary = open(os.path.join(summary_dir, summ)).read()
        doc = nlp(article)
        article_sents = " <split1> ".join([sent.text for sent in doc.sents])

        sent_res = get_sent_dist(summary, article_sents, minimum_seq=minimum_seq)

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

    return res

def get_avg_sent_len(summary_dir):
    summary_files = os.listdir(summary_dir)
    avg_sum_len = []
    for summary in tqdm(summary_files):
        summary = open(os.path.join(summary_dir, summary)).read()
        avg_sum_len.append(len(' '.join(summary.split('\n')).split()))

    return np.average(avg_sum_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--article_dir', default='test_art', help='article directory path')
    parser.add_argument('--summary_dir', default='test', help='summary directory path')
    parser.add_argument('--out_file_path', default='stats_summary.txt', help='output file path')
    parser.add_argument('--min_seq_len', type=int, default=3, help='minimum sequence length to consider a match')
    parser.add_argument('--file_suff', default='decoded', help='file suffix generally decode or reference')

    args = parser.parse_args()

    res = get_avg_sent_copied(args.article_dir, args.summary_dir, args.min_seq_len, args.file_suff)

    print(args)
    for key in res:
        print('{}: {}'.format(key, res[key]))

    with open(args.out_file_path, 'w') as out_file:
        out_file.write(str(args))
        for key in res:
            out_file.write('{}: {}\n'.format(key, res[key]))
