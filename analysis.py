import os
from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_web_lg")

from collections import Counter, defaultdict

def leaf_node_proportion(parent):
    n = len(parent)
    # Start with all nodes being leafs.
    is_leaf = [1 for i in range(n)]
    for i in range(1, n): # starts at -1 for the first artificial token.
        # Set the array to 0 for nodes that are someone's head.
        is_leaf[parent[i]] = 0
    return sum(is_leaf)/(len(is_leaf) - 1) # -1 for the first artificial node

# This functio fills depth of i'th element in parent[] 
# The depth is filled in depth[i]   
def fill_depth(parent, i , depth): 
      
    # If depth[i] is already filled 
    if depth[i] != 0: 
        return 
      
    # If node at index i is root 
    if parent[i] == -1: 
        depth[i] = 0
        return 
  
    # If depth of parent is not evaluated before, 
    # then evaluate depth of parent first 
    if depth[parent[i]] == 0: 
        fill_depth(parent, parent[i] , depth) 
  
    # Depth of this node is depth of parent plus 1 
    depth[i] = depth[parent[i]] + 1
  
# This function reutns height of binary tree represented 
# by parent array 
def find_height(parent): 
    n = len(parent)   
    # Create an array to store depth of all nodes and  
    # initialize depth of every node as 0 
    # Depth of root is 1 
    depth = [0 for i in range(n)] 
  
    # fill depth of all nodes 
    for i in range(n): 
        fill_depth(parent, i, depth) 
  
    # The height of binary tree is maximum of all  
    # depths. Find the maximum in depth[] and assign  
    # it to ht 
    ht = depth[0] 
    for i in range(1,n): 
        ht = max(ht, depth[i]) 
  
    return ht 

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
    ssplit = summary.split()  # .split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    seen_sentences = set()
    sentence_copy_id = defaultdict(set)
    
    for sent in summary.strip().split('.'):
        print(sent)
    sent_len = [len(sent.split()) for sent in summary.strip().split('.')]
    if 0 in sent_len:
        sent_len.remove(0)
    summary_sent_idx = [sum(sent_len[:i+1]) for i in range(len(sent_len))]
    print(summary_sent_idx)
    print(sent_len)
    
    # current_match_sidx = startix
    # current_match_eidx = endix
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
                # current_match_store[idx] = (startix, endix, endix-startix)
        if match_found:
            endix += 1
            longest_match_list = current_match_list
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix - 2:
                # matches.extend(["0"] * (endix - startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words

                full_string = compile_substring(startix, endix - 1, ssplit)
                if matchstrings[full_string] >= 1:
                    #matches.extend(["0"] * (endix - startix))
                    pass
                else:
                    #matches.extend(["1"] * (endix - startix))
                    #print(full_string)
                    seen_sentences.update(longest_match_list)
                    matchstrings[full_string] += 1

                # Extract the index of the sentence in the summary from which the subsequence was coming from.
                print(endix)
                print(summary_sent_idx)
                sentence_index = summary_sent_idx.index(min(i for i in summary_sent_idx if i > endix-1))
                # Save the sentence that it was coming from in the article.
                sentence_copy_id[sentence_index].update(longest_match_list)

                longest_match_list = []
                # endix += 1
            startix = endix
    
    avg_max_seq_len = None
    if len(matchstrings.values()) != 0:
        # List of susequence length times the number of occurrences of that susequence.
        tot_seq_len = [len(susequence.split()) * matchstrings[susequence] for susequence in matchstrings.keys()]
        # Get the average by summing and deviding by the total number of susequence matches.
        avg_max_seq_len = sum(tot_seq_len) / sum(matchstrings.values())

    # We want to calculate the average number of tokens that we copied from each original sentence.


    return list(seen_sentences), len(article), avg_max_seq_len

def get_avg_sent_copied(article_dir, summary_dir):
    article_files = os.listdir(article_dir)
    summary_files = os.listdir(summary_dir)
    sent_counter = []
    for article in tqdm(article_files):
        counter = article.split("_")[0]
        summ = str(counter)+"_decoded.txt"
        #print(article, summ)
        article = open(os.path.join(article_dir, article)).read()
        summary = open(os.path.join(summary_dir, summ)).read()
        doc = nlp(article)
        article_sents = " <split1> ".join([sent.text for sent in doc.sents])
        seen_sent, art_len, _ = get_sent_dist(summary, article_sents)
        sent_counter.append((seen_sent, art_len))
    percentages = [float(len(seen_sent))/float(sent_count) for seen_sent, sent_count in sent_counter]
    avg_percentage = sum(percentages)/float(len(percentages))
    nosents = [len(seen_sent) for seen_sent, sent_count in sent_counter]
    avg_nosents = sum(nosents)/float(len(nosents))
    return avg_percentage, avg_nosents



if __name__ == '__main__':
    # article_dir = "./test_output/articles"
    # baseline_dir = "./test_output/baseline"
    # pointgen_dir = "./test_output/pointer-gen"
    # pointgen_cov_dir = "./test_output/pointer-gen-cov"
    # print("Doing baseline")
    # baseline_avg_percent, baseline_avg_nosents = get_avg_sent_copied(article_dir, baseline_dir)
    # print("Doing pointgen")
    # pointgen_avg_percent, pointgen_avg_nosents = get_avg_sent_copied(article_dir, pointgen_dir)
    # print("Doing coverage")
    # pointgen_cov_avg_percent, pointgen_cov_avg_nosents = get_avg_sent_copied(article_dir, pointgen_cov_dir)
    # print("Baseline: "+str(baseline_avg_nosents)+" "+str(baseline_avg_percent))
    # print("Pointgen: "+str(pointgen_avg_nosents)+" "+str(pointgen_avg_percent))
    # print("Pointgen_Cov: "+str(pointgen_cov_avg_nosents)+" "+str(pointgen_cov_avg_percent))
    
    # article = "artidoro is this is the longest string . <split1> this is the longest there are also other words in the summary . <split1> is the longest string ."

    # summary = "this is a the longest string a string . hello hello other words in . artidoro is the . artidoro is this is . is this is the longest string ."
    # summary = "hello my friend ."

    # print(get_sent_dist(summary, article))

    print(leaf_node_number([-1, 0, 1, 1, 1, 1, 2, 3]))

    print(find_height([-1, 0, 1, 1, 1, 1, 2, 2]))
