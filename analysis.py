from collections import Counter

def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end + 1])

def get_sent_dist(summary, article):
    # tsplit = t.split()
    ssplit = summary  # .split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    seen_sentences = set()

    # current_match_sidx = startix
    # current_match_eidx = endix
    longest_match_list = []
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        match_found = False
        current_match_list = []
        for idx, sent in article:
            if searchstring in sent \
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
            if startix >= endix -1:
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
                    seen_sentences.update(longest_match_list)
                    matchstrings[full_string] += 1
                    longest_match_list = []
                # endix += 1
            startix = endix
    edited_matches = []
    for word, tag in zip(ssplit, matches):
        if word == '<split1>':
            edited_matches.append('<split1>')
        else:
            edited_matches.append(tag)
    return " ".join(edited_matches)


if __name__ == '__main__':
    article_dir = ''
    ref_dir = ''
    generated_summ_dir = ''
