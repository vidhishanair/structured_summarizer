import os
from tqdm import tqdm
import spacy
from collections import Counter, defaultdict

nlp = spacy.load("en_core_web_lg")

def compile_substring(start, end, split):
    """
    Return the substring starting at start and ending at end.
    """
    if start == end:
        return split[start]
    return " ".join(split[start:end + 1])

def get_sent_dist(summary, article):
    """
    Returns the number of sentences that were copied from the article,
    and the number of sentences in the article.

    Considers a match if there is a common sequence of at least 3 words.
    """
    # tsplit = t.split()
    article = [sent.split() for sent in article.split("<split1>")]
    ssplit = summary.replace('.', '').split() 
    startix = 0
    endix = 0
    matchstrings = Counter()
    seen_sentences = set()
    sentence_copy_id = defaultdict(set)
    sentence_copy_id_count = Counter()

    sent_len = [len(sent.split()) for sent in summary.strip().split('.')]
    if 0 in sent_len:
        sent_len.remove(0)
    summary_sent_idx = [sum(sent_len[:i+1]) for i in range(len(sent_len))]

    longest_match_list = []
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        match_found = False
        current_match_list = []
        for idx, sent in enumerate(article):
            if searchstring in " ".join(sent) \
                    and endix < len(ssplit) - 1:
                match_found = True
                current_match_list.append(idx)
        if match_found:
            endix += 1
            longest_match_list = current_match_list
        else:
            # only phrases of length 3+ words should be considered.
            if startix >= endix - 2:
                endix += 1
            else:
                full_string = compile_substring(startix, endix - 1, ssplit)
                if matchstrings[full_string] >= 1:
                    pass
                else:
                    seen_sentences.update(longest_match_list)
                    matchstrings[full_string] += 1

                # Extract the index of the sentence in the summary from which the subsequence was coming from.
                sentence_index = summary_sent_idx.index(min(i for i in summary_sent_idx if i > endix-1))
                # Save the sentence that it was coming from in the article.
                sentence_copy_id[sentence_index].update(longest_match_list)

                # Count the number of mentions for sentences of a given index.
                sentence_copy_id_count[sentence_index] += 1

                longest_match_list = []
                # endix += 1
            startix = endix

    avg_max_seq_len = None
    if len(matchstrings.values()) != 0:
        # List of susequence length times the number of occurrences of that susequence.
        tot_seq_len = [len(susequence.split()) * matchstrings[susequence] for susequence in matchstrings.keys()]
        # Get the average by summing and deviding by the total number of subsequence matches.
        avg_max_seq_len = sum(tot_seq_len) / sum(matchstrings.values())

    # We want to calculate the average number of tokens that we copied from each original sentence.


    return list(seen_sentences), len(article), avg_max_seq_len, sentence_copy_id_count

def get_avg_sent_copied(article_dir, summary_dir):
    article_files = os.listdir(article_dir)
    # summary_files = os.listdir(summary_dir)
    sent_counter = []
    avg_max_seq_len_list = []
    tot_sent_id_count = Counter()
    for article in tqdm(article_files):
        counter = article.split("_")[0]
        summ = str(counter)+"_decoded.txt"
        article = open(os.path.join(article_dir, article)).read()
        summary = open(os.path.join(summary_dir, summ)).read()
        doc = nlp(article)
        article_sents = " <split1> ".join([sent.text for sent in doc.sents])
        seen_sent, art_len, avg_max_seq_len, sent_id_count = get_sent_dist(summary, article_sents)
        sent_counter.append((seen_sent, art_len))
        if avg_max_seq_len is not None:
            avg_max_seq_len_list.append(avg_max_seq_len)
        tot_sent_id_count += sent_id_count
    percentages = [float(len(seen_sent))/float(sent_count) for seen_sent, sent_count in sent_counter]
    avg_percentage = sum(percentages)/float(len(percentages))
    nosents = [len(seen_sent) for seen_sent, sent_count in sent_counter]
    avg_nosents = sum(nosents)/float(len(nosents))
    avg_max_seq_len_tot = sum(avg_max_seq_len_list) / len(avg_max_seq_len_list)
    return avg_percentage, avg_nosents, avg_max_seq_len_tot, tot_sent_id_count


if __name__ == '__main__':
    article_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/articles"
    baseline_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/baseline"
    pointgen_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/pointer-gen"
    pointgen_cov_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/pointer-gen-cov"

    out_file_path = 'stats_summary.txt'

    print("Doing baseline")
    baseline_avg_percent, baseline_avg_nosents, baseline_avg_seq_len, baseline_tot_sent_id_count = get_avg_sent_copied(article_dir, baseline_dir)
    print("Doing pointgen")
    pointgen_avg_percent, pointgen_avg_nosents, pointgen_avg_seq_len, pointgen_tot_sent_id_count = get_avg_sent_copied(article_dir, pointgen_dir)
    print("Doing coverage")
    pointgen_cov_avg_percent, pointgen_cov_avg_nosents, pointgen_cov_avg_seq_len, pointgen_cov_tot_sent_id_count = get_avg_sent_copied(article_dir, pointgen_cov_dir)

    # Write to file.
    with open(out_file_path, 'w') as out_file:
        out_file.write("Baseline:\n")
        out_file.write("Average percentage of sentences copied: "+str(baseline_avg_percent) + "\n")
        out_file.write("Average count of sentences copied: "+str(baseline_avg_nosents)+"\n")
        out_file.write("Average length of matching subsequences: "+str(baseline_avg_seq_len)+"\n")
        out_file.write("Distribution over copied sentences id:\n")
        out_file.write(str(baseline_tot_sent_id_count))

        out_file.write("Pointgen:\n")
        out_file.write("Average percentage of sentences copied: "+str(pointgen_avg_percent) + "\n")
        out_file.write("Average count of sentences copied: "+str(pointgen_avg_nosents)+"\n")
        out_file.write("Average length of matching subsequences: "+str(pointgen_avg_seq_len)+"\n")
        out_file.write("Distribution over copied sentences id:\n")
        out_file.write(str(pointgen_tot_sent_id_count))

        out_file.write("Pointgen_Cov:\n")
        out_file.write("Average percentage of sentences copied: "+str(pointgen_cov_avg_percent) + "\n")
        out_file.write("Average count of sentences copied: "+str(pointgen_cov_avg_nosents)+"\n")
        out_file.write("Average length of matching subsequences: "+str(pointgen_cov_avg_seq_len)+"\n")
        out_file.write("Distribution over copied sentences id:\n")
        out_file.write(str(pointgen_cov_tot_sent_id_count))

    # Dump to stout.
    with open(out_file_path) as out_file:
        print(out_file.read())


    # article = "artidoro is this is the longest string . <split1> this is the longest there are also other words in the summary . <split1> is the longest string ."
    # summary = "this is a the longest string a string . hello hello other words in . artidoro is the . artidoro is this is . is this is the longest string ."
    # summary = "hello my friend ."
    # print(get_sent_dist(summary, article))
    # print(leaf_node_number([-1, 0, 1, 1, 1, 1, 2, 3]))
    # print(find_height([-1, 0, 1, 1, 1, 1, 2, 2]))
