from collections import Counter

def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end + 1])

def get_sent_dist(summary, article):
    # tsplit = t.split()
    article = [sent.split() for sent in article.split("<split1>")]
    ssplit = summary.split()  # .split()
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
                    print(full_string)
                    seen_sentences.update(longest_match_list)
                    matchstrings[full_string] += 1
                longest_match_list = []
                # endix += 1
            startix = endix

    return list(seen_sentences), len(article)


if __name__ == '__main__':
    article_dir = "every country in the world may have its rich kids of instagram , but in mexico the increasingly ostentatious displays of wealth by the young elite is taking on a political dimension . <split1> in a country where a significant majority of citizens live in poverty , the self-styled mirreyes -lrb- my kings -rrb- , with their champagne and sports cars , are increasingly exposing the amount of wealth concentrated into a relatively small number of hands . <split1> mirreyes , which refers to individuals who enjoy ` ostentatious spending , exhibitionism and narcissism ' and are ` placed above all others ' , post pictures of luxuries with a hashtag of the same name . <split1> scroll down for video . <split1> ballers and shotcallers : mirreyes are often the sons and daughters of the members of mexican high society . <split1> laying out and living it up : many of the photos posted by mirreyes showcase luxurious vacation spots . <split1> it 's just one big fiesta : a glance through the social media profile of a mirrey often reveals a party picture . <split1> bling : good clothes , nice watches and a flashy style are almost prerequisites for becoming one of the mirreyes . <split1> ready for takeoff : an expert on the subject of mirreyes said ` the key is the purchasing power the mirrey boasts ' <split1> three for the money : mirreyes ' displays of wealth can lead to political and legal advantages down the road . <split1> they are often the sons and daughters of government officials , wealthy businessman and other members of mexican high society and although their displays of wealth can be embarrassing , they can also lead to political and legal advantages , maclean 's reported . <split1> one such individual , the eldest son of then-chiapas attorney general raciel lopez salazar , jumped off a cruise ship off the coast of brazil during the 2014 world cup and vanished without a trace , . <split1> before he jumped off the ship 's 15th floor and disappeared beneath the waves , 29-year-old jorge alberto l\xc3\xb3pez amores told his friends to take cellphone videos of his leap . <split1> he said : ` i 'll stop this cruise , i 'll make history . ' <split1> l\xc3\xb3pez got his wish as the boat stopped for two hours to search for him , according to vivelo hoy . <split1> his story is not the only example of the celebration of excess gone wrong . <split1> in 2012 , the daughter of oil workers ' union boss carlos romero deschamps was brought into the spotlight for posting photos of her luxury lifestyle , including gucci bags and expensive wine , on facebook . <split1> her flashy pictures did not sit well in a country where people work an average of 2 226 hours a year and households take in an average of $ 12,850 per year after taxes . <split1> their stories are just examples of what the author of a book about mirreyes called the mexican ` generation me ' . <split1> keeping time : people often post pictures of watches and other luxuries like designer bags and or alcohol . <split1> rought : in mexico , the images of wealth are shocking as households take in an average of $ 12,850 per year . <split1> symbol : jorge alberto l\xc3\xb3pez amores -lrb- right -rrb- , who likely died after jumping off a boat , is an example of a mirrey . <split1> purchasing power : mirreyes might not even think twice about spending hundreds of dollars on a stuffed bear . <split1> selfie star : l\xc3\xb3pez , the son of then-chiapas attorney general raciel lopez salazar , jumped off a yacht in 2014 . <split1> no need to pimp my ride : mirreyes obtain cash from sources like inheritance , theft , corruption or the lottery . <split1> kiss the bling : many of the photos posted by mirreyes will either have a shiny watch , fancy champagne , or both . <split1> the author of mirreynato , the other inequality , ricardo raphael , wrote in his book : ` the economic wealth mirreyes use is the main marker of class . <split1> ` no matter where the money comes - work , inheritance , theft , corruption or lottery - the key is the purchasing power the mirrey boasts ' . <split1> by identifying themselves as mirreyes on social media , these seemingly spoiled brats are possibly creating connections which could be important for business and politics in the future because money attracts powerful friends . <split1> they are certainly being noticed . <split1> in 2011 , a a website called mirrreybook was created by pepe ceballos . <split1> he began posting photos of mmirreyes on the site and it quickly became popular . <split1> ceballos said : ` mirreyes , who do n't have anything to do with narcos -lsb- narcotics traffickers -rsb- , seem like they 're competing with them to see who has more . <split1> ` we 've suddenly seen it in the kind of fashion or the bling-bling . there 's more influence from the narcos . ' <split1> in a recent source of embarrassment , mirreyes were also featured in a controversial video that showcased the class of 2015 from private school instituto cumbres in mexico city . <split1> in the professionally-produced video , well-dressed male students were shown drinking alcohol while carrying out a casting call with young women who dance , strip and wash their feet . <split1> popping bottles : mirreyes are individuals who enjoy ` ostentatious spending , exhibitionism and narcissism ' <split1> the flash and the not-so-furious : members of this select group seem to have few reasons to be upset with life . <split1> up in smoke : some mirreyes post photos implying they have something in common with narcotics traffickers . <split1> look at me : the selfie has become somewhat of a status symbol and mirreyes frequently use them to show off . <split1> future endeavors : by identifying themselves on social media , mirreyes are possibly creating good connections . <split1> the all-male catholic school was accused of sexism and promoting privilege after the video began to spread across the internet , according to the national catholic reporter . <split1> the school issued an apology late last month . <split1> it said : ` the cumbres institute mexico asks for an apology for the content of the video that offended various persons , who have expressed anger . <split1> ` this video in no way represents the values and principles of the school , students , families and graduates . <split1> `` the necessary measures are being taken with the students involved and to establish rules so that it does not occur again . ' <split1> permanent vacation it seems mirreyes travel frequently to foreign destinations judging by the photos they post . <split1> power : mirreyes could be important for business and politics in the future because cash attracts friends . <split1> ouch : pics of mirreyes relaxing do n't sit well in a country where people work an average of 2 226 hours a year ."
    article_sents = [sent.split() for sent in article_dir.split("<split1>")]
    ref_dir = "the self-styled mirreyes refers to individuals who enjoy ` ostentatious spending , exhibitionism and narcissism ' and are ` placed above all others ' , post pictures of luxuries with a hashtag of the same name .  they are often the sons and daughters of government officials , wealthy businessman and other members of mexican high society ."
    print(get_sent_dist(ref_dir.split(), article_sents))
