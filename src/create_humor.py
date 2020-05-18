from generate_unconditional_samples import sample_combined_models, sample_model, sample_model_with_seed, \
    sample_combined_models_with_seed
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import XLNetTokenizer, XLNetLMHeadModel
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from nltk import sent_tokenize
from grammarbot import GrammarBotClient
import nltk

sentence_ending = ['.', '!', '?']
# words that we do not want XLNet to generate ever
exclude_words = ['<eop>', '<eod>']
# words that we do not want XLNet to generate when using idea2
avoid_xlnet_idea2 = ['.', ',', '?', '!', '<eop>', 'and', 'but', '<eop>', '<eod>']
# used in xlnet_fill_middle as a placeholder word
special_token = '<**>'
# words we do not want XLNet to generate
avoid_first = ['and', 'but']
avoid_last = ['and', 'but', 'so']


# Idea 1
# 1. Get Context Sentence from Domain 1 (GPT-2)
# 2. Find "pivot sentence" that is plausible from both domain 1 and domain 2 (maybe a 50/50 mix from both domains)
# 3. Use XLNet to fill in the sentence with different masks to go between domain 1 and domain 2
# 4. Generate text after pivot using domain 2 (GPT-2)
# Note we can potentially not show the original domain sentence to the end user
# Inputs:
# context1 : string run_name of the gpt-2 model
# context2 : string run_name of the gpt-2 model
# run_cnt: int used for file name printing
# fill_backwards1: A parameter that will have XLNet fill in the masks backwards
# fill_backward2: A parameter that will have XLNet fill in the masks backwards
# *** The reason that there are two fill_backwards is because of the split_two parameter. When split_two parameter is
#     true, we try to use XLNet twice.
#     fill_backwards1 is used for the first i//2 masks
#     fill_backwards2 is used for the last i//2 masks
# split_two: a testing parameter that will try to fill in i//2 masks at a time.
#         In other words it will try to fill in the first i//2 words of the next sentence first and then using those
#         filled in words, it will then try to fill in the rest of the i//2 masks with the context2 sentence at the end
# use_gpt2 : a testing parameter that when true, will use gpt-2 to fill in the first i words
#            of the sentence to be created by XLNet in step 3
def idea1(context1, context2, run_cnt, fill_backwards1=True, fill_backwards2=False, split_two=False, use_gpt2=True):
    # 1. Get Context Sentence from Domain 1 (GPT-2)
    # context_sent1 = 'John and Sue walked together on the beach'
    if split_two:
        if fill_backwards1 and fill_backwards2:
            f = open('idea1_samples/backward_fill/split_two/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1,
                                                                                                     context2), 'w',
                     encoding='utf-8')
        elif not fill_backwards1 and fill_backwards2:
            f = open(
                'idea1_samples/first_forward_second_backward/split_two/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt,
                                                                                                                context1,
                                                                                                                context2),
                'w', encoding='utf-8')
        elif fill_backwards1 and not fill_backwards2:
            f = open(
                'idea1_samples/first_backward_second_forward/split_two/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt,
                                                                                                                context1,
                                                                                                                context2),
                'w', encoding='utf-8')
        else:
            f = open('idea1_samples/forward_fill/split_two/scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1,
                                                                                                    context2), 'w',
                     encoding='utf-8')
    if use_gpt2:
        if fill_backwards1 and fill_backwards2:
            f = open(
                'idea1_samples/backward_fill/gpt2/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1, context2),
                'w', encoding='utf-8')
        elif not fill_backwards1 and fill_backwards2:
            f = open('idea1_samples/first_forward_second_backward/gpt2/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt,
                                                                                                                context1,
                                                                                                                context2),
                     'w', encoding='utf-8')
        elif fill_backwards1 and not fill_backwards2:
            f = open('idea1_samples/first_backward_second_forward/gpt2/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt,
                                                                                                                context1,
                                                                                                                context2),
                     'w', encoding='utf-8')
        else:
            f = open(
                'idea1_samples/forward_fill/gpt2/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1, context2),
                'w', encoding='utf-8')
    else:
        if fill_backwards1 and fill_backwards2:
            f = open('idea1_samples/backward_fill/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1, context2),
                     'w', encoding='utf-8')
        elif not fill_backwards1 and fill_backwards2:
            f = open(
                'idea1_samples/first_forward_second_backward/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1,
                                                                                                      context2), 'w',
                encoding='utf-8')
        elif fill_backwards1 and not fill_backwards2:
            f = open(
                'idea1_samples/first_backward_second_forward/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1,
                                                                                                      context2), 'w',
                encoding='utf-8')
        else:
            f = open('idea1_samples/forward_fill/lm_scoring_{}_idea1_{}_{}_out.txt'.format(run_cnt, context1, context2),
                     'w', encoding='utf-8')
    isSent = False
    while not isSent:
        context_sent1 = sample_model(
            model_name='117M',
            run_name=context1,
            seed=None,
            nsamples=1,
            batch_size=1,
            length=60,
            temperature=1,
            top_k=40,
            top_p=0.0)
        if len(sent_tokenize(context_sent1)) >= 2:
            context_sent1 = ' '.join(sent_tokenize(context_sent1)[0:2])
        else:
            context_sent1 = sent_tokenize(context_sent1)[0]
        # print('context_sent1: {}'.format(context_sent1))
        isSent = isSentence(context_sent1)
    context_sent1 += ' '
    context_sent1 = context_sent1.replace('\n', '')
    # 2. Find "pivot sentence" that is plausible from both domain 1 and domain 2 (maybe a 50/50 mix from both domains)
    # pivot_sentence = 'John got down on one knee'
    isSent = False
    while not isSent:
        pivot_sentence = sample_combined_models_with_seed(
            model_name='117M',
            run_name1=context1,
            run_name2=context2,
            seed=None,
            nsamples=1,
            batch_size=1,
            length=50,
            temperature=1,
            top_k=40,
            top_k_combined=0.0,
            top_p=0.0,
            weight1=0.5,
            weight2=0.5,
            use_random=False,
            use_swap=False,
            use_fifty_one=False,
            use_vanilla=False,
            debug=False,
            raw_text=context_sent1)
        pivot_sentence = sent_tokenize(pivot_sentence)[0]
        isSent = isSentence(pivot_sentence)
    isSent = False
    pivot_sentence = pivot_sentence.replace('\n', '')
    # 3. Generate text after pivot using domain 2 (GPT-2)
    while not isSent:
        context_sent2 = sample_model_with_seed(
            model_name='117M',
            run_name=context2,
            seed=None,
            nsamples=1,
            batch_size=1,
            length=50,
            temperature=1,
            top_k=40,
            top_p=0.0,
            raw_text=pivot_sentence)
        if len(sent_tokenize(context_sent2)) >= 2:
            context_sent2 = ' '.join(sent_tokenize(context_sent2)[0:2])
        else:
            context_sent2 = sent_tokenize(context_sent2)[0]
        isSent = isSentence(context_sent2)
    context_sent2 = context_sent2.replace('\n', '')
    # 4. Use XLNet to fill in the sentence with different masks to go between domain 1 and domain 2
    # Try different mask lengths and choose the "best" one
    #print('context 1: {}'.format(context_sent1))
    #print('context 2: {}'.format(context_sent2))
    best_sent = ''
    best_val = 100
    best_num_masks = 0
    start = 4
    end = 20
    best_before = ''
    # Iterating between start and end number of masks to see which generation is the best
    for i in range(start, end):
        # Here we use gpt-2 to generate the first i words of the sentence
        if use_gpt2:
            orig_sent1 = context_sent1 + pivot_sentence + ' '
            is_good = False
            while not is_good:
                first_half = sample_model_with_seed(model_name='117M', run_name='model', seed=None, nsamples=1,
                                                    batch_size=1, length=i, temperature=1, top_k=40, top_p=0.0,
                                                    raw_text=orig_sent1)
                is_good = 'www.' not in first_half
            orig_sent1 = orig_sent1 + first_half + ' <mask> ' * (i // 2) + context_sent2
            #print('first part: {}'.format(orig_sent1))
            f.write('\nfirst part: {}\n'.format(first_half))
            f.write('orig_sent:{}\n'.format(orig_sent1))
            before = orig_sent1
            out_sent, masked_out1 = xl_net_fill_middle(orig_sent1, fill_backwards=fill_backwards1)
            #print('\nout_sent: {}\n'.format(out_sent))
            f.write('out_sent: {}\n'.format(out_sent))
            masked_part = first_half + masked_out1
            score = idea1_score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_part, context_sent2)
        # split_two was a testing parameter that will try to fill in i//2 masks at a time.
        # in other words it will try to fill in the first i//2 words of the next sentence first and then using those
        # filled in words, it will then try to fill in the rest of the i//2 masks with the context2 sentence at the end
        elif split_two:
            before = context_sent1 + pivot_sentence + ' <mask> ' * i + context_sent2
            orig_sent1 = context_sent1 + pivot_sentence + ' <mask> ' * (
                        i // 2)  # + ' '.join(context2_words[:split_len])
            # orig_sent2 = ' <mask> '*(i//2) + '.' + context_sent
            out_sent = ''
            print('{} masks'.format(i))
            f.write('\n{} masks\n'.format(i))
            print('\norig_sent1: {}\n'.format(orig_sent1))
            f.write('orig_sent1: {}\n'.format(orig_sent1))
            out1, masked_out1 = xl_net_fill_middle(orig_sent1, fill_backwards=fill_backwards1)
            print('\nout_sent1: {}\n'.format(out1))
            f.write('out_sent1: {}\n'.format(out1))
            out_sent += out1
            # second_part = sent_tokenize(orig_sent1)[-1]
            orig_sent2 = out1 + ' <mask> ' * (i // 2) + '.' + context_sent2
            print('\norig_sent2: {}\n'.format(orig_sent2))
            f.write('orig_sent2: {}\n'.format(orig_sent2))
            out2, masked_out2 = xl_net_fill_middle(orig_sent2, fill_backwards=fill_backwards2)
            print('\nout_sent2: {}\n'.format(out2))
            f.write('out_sent2: {}\n'.format(out2))
            out_sent = out2
            score = idea1_score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_out2, context_sent2)
        # Otherwise we call xl_net_fill_middle once to try to fill in the i <mask>s.
        else:
            orig_sent1 = context_sent1 + pivot_sentence + ' <mask> ' * i + context_sent2
            before = orig_sent1
            print('{} masks'.format(i))
            f.write('\n{} masks\n'.format(i))
            f.write('orig: {}'.format(orig_sent1))
            out_sent, masked_out = xl_net_fill_middle(orig_sent1, fill_backwards=fill_backwards1)
            print('\nout_sent2: {}\n'.format(out_sent))
            f.write('out_sent2: {}\n'.format(out_sent))
            score = idea1_score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_out, context_sent2)
        # choose the best score
        if score > best_val or i == start:
            best_sent = out_sent
            best_val = score
            best_num_masks = i
            best_before = before
    f.write('\n\n')
    f.write('\nBest Output\n')
    f.write('num masks: {}\n'.format(best_num_masks))
    f.write('\nwith Masks: {}\n'.format(best_before))
    f.write('\n\nbest: {}\n'.format(best_sent))
    f.write('score: {}'.format(best_val))
    f.close()
    # TODO Note we can potentially not show the original domain sentence to the end user
    return best_sent


# Scores the grammar using a Grammar Client
# Recently I have been favoring the LM Scoring method
def score_sentence_grammar(context_sent1, pivot_sentence, masked_sentence, context_sent2):
    client = GrammarBotClient()
    sentence = context_sent1 + pivot_sentence + masked_sentence + context_sent2
    res = client.check(sentence)
    score = len(res.matches)
    return score


# Score the sentence based on LM Scoring method
# Inputs are the different parts of the output created by Idea1.
# Here we use a similar method of scoring as the Conversational Humor Reward function using sentiment analysis
# https://github.com/kcaras/ConversationalHumor
def idea1_score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_sentence, context_sent2):
    pivot = get_sentiment(pivot_sentence)
    masked_sent = get_sentiment(masked_sentence)
    sent2 = get_sentiment(context_sent2)
    len_ex = sum([len(text.split()) for text in [context_sent1, pivot_sentence, masked_sentence, context_sent2]])
    score = 0
    sentiment_difference1 = sent2 - masked_sent
    sentiment_difference2 = masked_sent - pivot
    if sentiment_difference1 < 0.0 and sent2 >= 0.0:
        sentiment_difference1 = 0.0
    if sentiment_difference2 < 0.0 and masked_sent >= 0.0:
        sentiment_difference2 = 0.0
    # need to figure out a score with sentiment
    # if (masked_sent > 0 and sent2 < 0) or (sent2 > 0 and masked_sent < 0):
    #    score += 1
    # if (pivot > 0 and sent2 < 0) or (sent2 > 0 and pivot < 0):
    #    score += 1
    grammar_score = sum([lm_scoring(text) for text in [context_sent1, pivot_sentence, masked_sentence, context_sent2]])/4.0
    score = (sentiment_difference1 + sentiment_difference2 - grammar_score) / 4
    return score


# Alternative scoring method for idea 1. (Older, tend to favor idea1_score_sentence_lm_scoring method)
# Attempts to combine the grammar + sentiment scoring methods in ConversationalHumor project
# Inputs are the different sentences produced by Idea 1
def idea1_score_sentence_grammar_sentiment(context_sent1, pivot_sentence, masked_sentence, context_sent2):
    sent1 = get_sentiment(context_sent1)
    pivot = get_sentiment(pivot_sentence)
    masked_sent = get_sentiment(masked_sentence)
    sent2 = get_sentiment(context_sent2)
    len_ex = sum([len(text.split()) for text in [context_sent1, pivot_sentence, masked_sentence, context_sent2]])
    score = 0
    sentiment_difference1 = sent2 - masked_sent
    sentiment_difference2 = masked_sent - pivot
    if sentiment_difference1 < 0.0 and sent2 >= 0.0:
        sentiment_difference1 = 0.0
    if sentiment_difference2 < 0.0 and masked_sent >= 0.0:
        sentiment_difference2 = 0.0
    # need to figure out a score with sentiment
    # if (masked_sent > 0 and sent2 < 0) or (sent2 > 0 and masked_sent < 0):
    #    score += 1
    # if (pivot > 0 and sent2 < 0) or (sent2 > 0 and pivot < 0):
    #    score += 1
    grammar_score = 0
    grammar_score = score_sentence_grammar(context_sent1, pivot_sentence, masked_sentence, context_sent2) / len_ex
    score = (sentiment_difference1 + sentiment_difference2 - grammar_score) / 4
    return score


# Method that determines if GPT-2/XLNet output is an acceptable sentence
# sentence: string
# at_least_length: integer the minimum lenght of the sentence
def isSentence(sentence, at_least_length=10):
    has_word = len(sentence.split()) > 0
    sentence = sentence.strip()
    return has_word and sentence != '' and any([sentence[-1] == punct for punct in sentence_ending]) and len(
        sentence.split()) > at_least_length and 'www.' not in sentence and 'http' not in sentence and '/r' not in sentence


# Method that determines if the
# sb: string "sentence beginning"
# at_least_length: the minimum lenght of the sentence beginning
def isSentenceBeginning(sb, at_least_length=3):
    sb = sb.strip()
    return all([sb[-1] != punct for punct in sentence_ending]) and len(sb.split()) > at_least_length


# Idea 2:
# TODO Important note! Recently I have been making the word in step 1 the same as the related word (step 3).
# TODO Please see main_idea2 for inputs
# 1. Find a word with multiple senses (word)
# 2. Generate sentence from domain 1 (GPT-2) that uses that word
# 3. Find another word from domain 2 that is related to the word (related_word)
# 4. Generate sentence using that related word.
# 5. Find a word used only in domain 2
# 6. Generate a sentence in domain 2 (either contains only the word in step 5 or the word in step 5 and the word in step 4)
# Inputs
# context1 : string run_name of the gpt-2 model
# context2 : string run_name of the gpt-2 model
# word: string word with multiple senses that can fit in both domain 1 and domain 2
# related_word: string word that is related to word but can also fit in domain 2
# word2: string word that generally is only used in domain 2
# run_cnt: int used for file name printing
# sentence_len: the maxium length of each sentence generated by step 2, and 4
# two_words: two_words is a parameter that will fill in the last sentence using both the related word and the word used only in domain 2
# remove_middle: Remove_middle is a parameter that will discard the sentence generated in step 4
#                which is the sentence in domain 2 that used only the related word
def idea2(context1, context2, word='', related_word='', word2='', run_cnt=0, sentence_len=30, two_words=True, remove_middle=False):
    # 1. Find a word with multiple senses
    if word == '':
        word = 'gifted'
    else:
        print('word: {}'.format(word))

    isSent = False
    if two_words and remove_middle:
        f = open('idea2_samples/lm_scoring/two_words/remove_middle/idea2_{}_len_{}_{}_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1, context2, word, related_word, word2),
             'w', encoding='utf-8')
    elif two_words:
        f = open('idea2_samples/lm_scoring/two_words/idea2_{}_len_{}_{}_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1, context2, word, related_word, word2),'w', encoding='utf-8')
    elif remove_middle:
        f = open('idea2_samples/lm_scoring/remove_middle_only/idea2_{}_len_{}_{}_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1, context2, word, related_word, word2),'w', encoding='utf-8')
    else:
        f = open('idea2_samples/lm_scoring/idea2_{}_len_{}_{}_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1, context2, word, related_word, word2),
             'w', encoding='utf-8')
    f.write('word: {}\nrelated_word: {}\nword2:{}\n'.format(word, related_word, word2))
    # Generate a throw away sentence from context 1 to start out the XLNet
    while not isSent:
        context_sent1_throw = sample_model(model_name='117M',
                                           run_name=context1,
                                           seed=None,
                                           nsamples=1,
                                           batch_size=1,
                                           length=sentence_len,
                                           temperature=1,
                                           top_k=40,
                                           top_p=0.0)
        context_sent1_throw = sent_tokenize(context_sent1_throw)
        context_sent1_throw = ' '.join(context_sent1_throw[:-1])
        isSent = isSentence(context_sent1_throw)

    #print('context_sent1_throw: {}'.format(context_sent1_throw))
    f.write('context_sent1_throw: {}\n'.format(context_sent1_throw))
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    sentence_beginning1 = xl_net_fill_begining(context_sent1_throw, word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    # sentence_beginning1 = 'He {} '.format(word)
    #print('sentence_beginning1: {}'.format(sentence_beginning1))
    f.write('\nsentence_beginning1: {}\n'.format(sentence_beginning1))
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    isSent = False
    while not isSent:
        context_sent1 = sample_model_with_seed(model_name='117M',
                                               run_name=context1,
                                               seed=None,
                                               nsamples=1,
                                               batch_size=1,
                                               length=sentence_len,
                                               temperature=1,
                                               top_k=40,
                                               top_p=0.0,
                                               raw_text=sentence_beginning1)
        context_sent1 = sent_tokenize(sentence_beginning1 + context_sent1)
        context_sent1 = ' '.join(context_sent1[:-1])
        isSent = isSentence(context_sent1)
    #print('context_sent1: {}'.format(context_sent1))
    f.write('\ncontext_sent1: {}\n'.format(context_sent1))
    # 3. Find another word from domain 2 that is related to the word (maybe using word vectors or frequencies???)
    if related_word == '':
        related_word = 'child'
    else:
        print('related_word: {}'.format(related_word))

    # Generate a throw away sentence from context 2 to start out the XLNet
    isSent = False
    while not isSent:
        context_sent2_throw = sample_model(model_name='117M',
                                           run_name=context2,
                                           seed=None,
                                           nsamples=1,
                                           batch_size=1,
                                           length=sentence_len,
                                           temperature=1,
                                           top_k=40,
                                           top_p=0.0)
        context_sent2_throw = sent_tokenize(context_sent2_throw)
        context_sent2_throw = ' '.join(context_sent2_throw[:-1])
        isSent = isSentence(context_sent2_throw)
    #print('context_sent2_throw: {}'.format(context_sent2_throw))
    sentence_beginning2 = xl_net_fill_begining(context_sent2_throw, related_word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    #print('sentence_beginning2: {}'.format(sentence_beginning2))

    # 4. Generate sentence using that related word in domain 2.
    isSent = False
    while not isSent:
        context_sent2 = sample_model_with_seed(model_name='117M',
                                               run_name=context2,
                                               seed=None,
                                               nsamples=1,
                                               batch_size=1,
                                               length=sentence_len,
                                               temperature=1,
                                               top_k=50,
                                               top_p=0.0,
                                               raw_text=context_sent1 + ' ' + sentence_beginning2
                                               )
        context_sent2 = sent_tokenize(sentence_beginning2 + context_sent2)
        context_sent2 = ' '.join(context_sent2[:-1])
        isSent = isSentence(context_sent2)
    #print('context_sent2: {}'.format(context_sent2))

    # 5. Find word only used in domain 2
    if word2 == '':
        word2 = 'pancake'
    else:
        print('word2 {}'.format(word2))
    # Generate a throw away sentence from context 2 to start out the XLNet
    isSent = False
    while not isSent:
        context_sent2_throw2 = sample_model(model_name='117M',
                                            run_name=context2,
                                            seed=None,
                                            nsamples=1,
                                            batch_size=1,
                                            length=sentence_len,
                                            temperature=1,
                                            top_k=40,
                                            top_p=0.0)
        context_sent2_throw2 = sent_tokenize(context_sent2_throw2)
        context_sent2_throw2 = ' '.join(context_sent2_throw2[:-1])
        isSent = isSentence(context_sent2_throw2)
    #print('context_sent2_throw2: {}'.format(context_sent2_throw2))

    # two_words is a parameter that will fill in the last sentence using both the related word and the word used only in domain 2
    if two_words:
        sentence_beginning3 = xl_net_fill_begining_two_words(context_sent2_throw2, related_word, word2, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    # otherwise just create a sentence in domain 2 that only contains word2 (word in domain 2 only)
    else:
        sentence_beginning3 = xl_net_fill_begining(context_sent2_throw2, word2, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    #print('sentence_beginning3: {}'.format(sentence_beginning3))
    isSent = False
    # Remove_middle is a parameter that will discard the sentence generated in step 4
    # which is the sentence in domain 2 that used only the related word
    if remove_middle:
        context_sent3_input = context_sent1 + ' ' + sentence_beginning3
    else:
        context_sent3_input = context_sent1 + ' ' + context_sent2 + ' ' + sentence_beginning3
    # 6. Generate a sentence in domain 2 (either contains only the word in step 5 or the word in step 5 and the word in step 4)
    while not isSent:
        context_sent3 = sample_model_with_seed(model_name='117M',
                                               run_name=context2,
                                               seed=None,
                                               nsamples=1,
                                               batch_size=1,
                                               length=sentence_len,
                                               temperature=1,
                                               top_k=50,
                                               top_p=0.0,
                                               raw_text=context_sent3_input)
        context_sent3 = sent_tokenize(sentence_beginning3 + context_sent3)
        context_sent3 = ' '.join(context_sent3[:-1])
        isSent = isSentence(context_sent3)

    print('context_sent3: {}'.format(context_sent3))
    print('\n\n')
    print('sentence_beginning1: {}'.format(sentence_beginning1))
    print('context_sent1: {}'.format(context_sent1))
    if remove_middle:
        print('***REMOVED Context Sent2')
    print('sentence_beginning2: {}'.format(sentence_beginning2))
    print('context_sent2: {}'.format(context_sent2))
    print('sentence_beginning3: {}'.format(sentence_beginning3))
    print('context_sent3: {}'.format(context_sent3))
    print('context_sent3: {}'.format(context_sent3))
    f.write('\n\n')
    f.write('sentence_beginning1: {}\n'.format(sentence_beginning1))
    f.write('context_sent1: {}\n'.format(context_sent1))
    if remove_middle:
        f.write('***Did not use Context Sent2***\n')
    f.write('sentence_beginning2: {}\n'.format(sentence_beginning2))
    f.write('context_sent2: {}\n'.format(context_sent2))
    f.write('\n\n')
    f.write('sentence_beginning3: {}\n'.format(sentence_beginning3))
    f.write('context_sent3: {}\n'.format(context_sent3))
    f.write('\n\n\n')
    if remove_middle:
        f.write(context_sent1 + ' ' + context_sent3)
    else:
        f.write(context_sent1 + ' ' + context_sent2 + ' ' + context_sent3)
    f.close()
    if remove_middle:
        return context_sent1 + ' ' + context_sent3
    else:
        return context_sent1 + ' ' + context_sent2 + ' ' + context_sent3


# Idea 2 is modified to use only one word with multiple senses.
# 1. Find a word with multiple senses (word)
# 2. Generate sentence from domain 1 (GPT-2) that uses that word
# 3a. Could possibly generate lead sentence in domain 2
# 3. Generate sentence from domain 2 (GPT-2) that uses that word
# 4. Generate sentence using that related word.
# inputs
# context1 : string run_name of the gpt-2 model
# context2 : string run_name of the gpt-2 model
# word: string word with multiple senses that can fit in both domain 1 and domain 2
# run_cnt: int used for file name printing
# sentence_len: the maximum length of each sentence generated by step 2, and 4
# use_leading: boolean use_leading is a parameter that generates a sentence from domain 2
#              The hope is that by using another sentence from domain 2, the context2 sentence will be much more
#              from domain 2 as apposed to being influenced too much by what was generated in context1.
# use_combo: boolean
def idea2_modified(context1, context2, word='', run_cnt=0, sentence_len=40, use_leading=True, use_combo=False):
    # 1. Find a word with multiple senses
    if word == '':
        word = 'gifted'
    else:
        print('word: {}'.format(word))

    isSent = False
    if use_leading and use_combo:
        f = open(
            'idea2_one_word_samples/use_leading/use_combo/idea2_{}_len_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1,
                                                                                     context2, word), 'w',
            encoding='utf-8')
    elif use_leading:
        f = open(
            'idea2_one_word_samples/use_leading/idea2_{}_len_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1,
                                                                                     context2, word), 'w',
            encoding='utf-8')
    elif use_combo:
        f = open(
            'idea2_one_word_samples/use_combo/idea2_{}_len_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1,
                                                                                     context2, word), 'w',
            encoding='utf-8')
    else:
        f = open('idea2_one_word_samples/idea2_{}_len_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1, context2,
                                                                              word), 'w', encoding='utf-8')
    # Generate a throw away word to start out the XLNet
    # Used to generate a sentence beginning for context 1
    while not isSent:
        context_sent1_throw = sample_model(model_name='117M',
                                           run_name=context1,
                                           seed=None,
                                           nsamples=1,
                                           batch_size=1,
                                           length=sentence_len,
                                           temperature=1,
                                           top_k=40,
                                           top_p=0.0)
        context_sent1_throw = sent_tokenize(context_sent1_throw)
        context_sent1_throw = ' '.join(context_sent1_throw[:-1])
        isSent = isSentence(context_sent1_throw)
    #print('context_sent1_throw: {}'.format(context_sent1_throw))
    sentence_beginning1 = xl_net_fill_begining(context_sent1_throw, word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)

    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    isSent = False
    while not isSent:
        context_sent1 = sample_model_with_seed(model_name='117M',
                                               run_name=context1,
                                               seed=None,
                                               nsamples=1,
                                               batch_size=1,
                                               length=sentence_len,
                                               temperature=1,
                                               top_k=40,
                                               top_p=0.0,
                                               raw_text=sentence_beginning1)
        context_sent1 = sent_tokenize(sentence_beginning1 + context_sent1)
        context_sent1 = ' '.join(context_sent1[:-1])
        isSent = isSentence(context_sent1)

    # Generate a throw away word to start out the XLNet
    # Used to generate a sentence beginning for context 2
    isSent = False
    while not isSent:
        context_sent2_throw = sample_model(model_name='117M',
                                           run_name=context2,
                                           seed=None,
                                           nsamples=1,
                                           batch_size=1,
                                           length=sentence_len,
                                           temperature=1,
                                           top_k=40,
                                           top_p=0.0)
        context_sent2_throw = sent_tokenize(context_sent2_throw)
        context_sent2_throw = ' '.join(context_sent2_throw[:-1])
        isSent = isSentence(context_sent2_throw)
    print('context_sent2_throw: {}'.format(context_sent2_throw))

    # 3. Generate sentence from domain 2 (GPT-2) that uses that word
    sentence_beginning2 = xl_net_fill_begining(context_sent2_throw, word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    print('sentence_beginning2: {}'.format(sentence_beginning2))
    context2_input = ''
    # use_leading is a parameter that generates a sentence from domain 2
    # The hope is that by using another sentence from domain 2, the context2 sentence will be much more from domain 2
    # as apposed to being influenced too much by what was generated in context1.
    if use_leading:
        # 5. Find word only used in domain 2
        isSent = False
        while not isSent:
            context_sent2_throw2 = sample_model(model_name='117M',
                                                run_name=context2,
                                                seed=None,
                                                nsamples=1,
                                                batch_size=1,
                                                length=sentence_len,
                                                temperature=1,
                                                top_k=40,
                                                top_p=0.0)
            context_sent2_throw2 = sent_tokenize(context_sent2_throw2)
            context_sent2_throw2 = ' '.join(context_sent2_throw2[:-1])
            isSent = isSentence(context_sent2_throw2)
        print('context_sent2_throw2: {}'.format(context_sent2_throw2))
        context2_input = context_sent1 + ' ' + context_sent2_throw2 + ' ' + sentence_beginning2
    # otherwise, just use the previous generated sentence as the input
    else:
        context2_input = context_sent1 + ' ' + sentence_beginning2

    # 3. Generate sentence using that word in domain 2.
    isSent = False
    # use_combo is a parameter that will use a combined model of both context1 and context2 to generate a sentence that
    # uses the word with multiple senses. In general, I found that did not produce funny results.
    # Right now the combiation is 0.3 context 1 and 0.7 context 2, but I also tried with 50/50.
    if use_combo:
        while not isSent:
            context_sent2 = sample_combined_models_with_seed(
                model_name='117M',
                run_name1=context1,
                run_name2=context2,
                seed=None,
                nsamples=1,
                batch_size=1,
                length=50,
                temperature=1,
                top_k=40,
                top_k_combined=0.0,
                top_p=0.0,
                weight1=0.3,
                weight2=0.7,
                use_random=False,
                use_swap=False,
                use_fifty_one=False,
                use_vanilla=False,
                debug=False,
                raw_text=context2_input)
            context_sent2 = sent_tokenize(sentence_beginning2 + context_sent2)
            context_sent2 = ' '.join(context_sent2[:-1])
            isSent = isSentence(context_sent2)
    # otherwise, just generate a context2 sentence using a 100% context 2 model.
    else:
        while not isSent:
            context_sent2 = sample_model_with_seed(model_name='117M',
                                                   run_name=context2,
                                                   seed=None,
                                                   nsamples=1,
                                                   batch_size=1,
                                                   length=sentence_len,
                                                   temperature=1,
                                                   top_k=50,
                                                   top_p=0.0,
                                                   raw_text=context2_input
                                                   )
            context_sent2 = sent_tokenize(sentence_beginning2 + context_sent2)
            context_sent2 = ' '.join(context_sent2[:-1])
            isSent = isSentence(context_sent2)

    print('context_sent2: {}'.format(context_sent2))
    print('\n\n')
    print('sentence_beginning1: {}'.format(sentence_beginning1))
    print('context_sent1: {}'.format(context_sent1))
    print('sentence_beginning2: {}'.format(sentence_beginning2))
    print('context_sent2: {}'.format(context_sent2))
    if use_leading:
        print('context2_sent_throw2:  {}'.format(context_sent2_throw2))
        f.write('context2_sent_throw2:  {}'.format(context_sent2_throw2))
    f.write('\n\n')
    f.write('sentence_beginning1: {}\n'.format(sentence_beginning1))
    f.write('context_sent1: {}\n'.format(context_sent1))
    f.write('sentence_beginning2: {}\n'.format(sentence_beginning2))
    f.write('context_sent2: {}\n'.format(context_sent2))
    f.write('\n\n\n')
    f.write(context_sent1 + ' ' + context_sent2)
    f.close()
    return context_sent1 + ' ' + context_sent2


# method that gets the sentiment of the sentence using NLTK Sentiment Analyzer
# sentence: string
def get_sentiment(sentence):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    return sentiment_analyzer.polarity_scores(sentence)['compound']


# Method that uses XLNet to create a sentence beginnninng using the two inputted words
# context_sent_throw: string a sentence that was generated from the desired context that is used to make sure
#                     XLNet starts with the desired context.
# word1: string first word you want to appear in the generated sentence beginning
# word2: string second word that you want to appear in the generated sentence beginning
#           **<!> Note word1 and word2 could appear in any order in the sentence beginning**
# start: integer the minimum number of <masks>
# end: integer the max number of <masks>
# k: integer topK used for XLNet output
# avoid: a list of words you want to avoid generating in the sentence beginning
def xl_net_fill_begining_two_words(context_sent_throw, word1, word2, start=1, end=6, k=5, avoid=None):
    # setting up XLNet
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
    output = []
    last_word = context_sent_throw.split()[-1]
    # Trying sentences with word1 appearing first and sentences with word2 appearing first
    for w1, w2 in [(word1, word2), (word2, word1)]:
        # Iterating over different number of <mask>
        for num_masks in range(start, end):
            # Use ((num_masks+1)//2) in between word1 and word2
            orig_sent = context_sent_throw + ' <mask> ' * num_masks + w1 + ' <mask> ' * ((num_masks+1)//2) + w2 + ' <mask>' * num_masks
            # begin to fill in the masks
            while '<mask>' in orig_sent:
                replacements = [ix for ix, word in enumerate(orig_sent.split()) if word == '<mask>']
                input_ids = torch.tensor(tokenizer.encode(orig_sent, add_special_tokens=True)).unsqueeze(0)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
                masked = (input_ids == 6).float()
                perm_mask = perm_mask + masked
                predicts = torch.nonzero(masked[0]).tolist()
                target_mapping = torch.zeros((1, len(predicts), input_ids.shape[1]), dtype=torch.float)
                out_sent = orig_sent.split()
                for n, p in enumerate(predicts):
                    target_mapping[0][n][p] = 1.0

                outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
                next_token_logits = outputs[
                    0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
                # Note that we only fill forward here because this is the beginning of the sentence.
                ranger = list(range(len(predicts)))
                getting_repeats = False
                # Here we will only look at one <mask> result at a time. This means that we will re-run XLNet after
                # we fill in the current <mask>
                for i in ranger[0:1]:
                    vals, idxs = torch.topk(next_token_logits[0][i], k)
                    word_list = [tokenizer.decode(idx) for idx in idxs.tolist()]
                    found = False
                    found_ix = 0
                    if getting_repeats:
                        out_sent[replacements[i]] = ''
                    else:
                        # Here we are iterating over the topk results for the first <mask> that we have not filled in already
                        while not found and found_ix < k - 1:
                            # pw ==> possible word
                            pw = word_list[found_ix]
                            # if the possible word is not one we want to avoid and it is not the same word as the previous word
                            if pw not in avoid and replacements[i] - 1 > 0 and replacements[i] - 1 < len(out_sent) and out_sent[
                                replacements[i] - 1].lower() != pw.lower():
                                found = True
                            # seen the same word again
                            elif replacements[i] - 1 < len(out_sent) and out_sent[replacements[i] - 1].lower() == pw.lower():
                                getting_repeats = True
                                found = True
                            # if this is the first word that we found
                            elif replacements[i] == 0 and pw not in avoid:
                                found = True
                            # if this is the last word and it is not a punctuation mark
                            # Note we are trying to generate a sentence beginning that will be filled in later by GPT-2
                            elif replacements[i] == len(out_sent) - 1 and pw not in sentence_ending:
                                found = True
                            else:
                                found_ix += 1
                        # if we are repeating too much, we just append the empty string instead of adding more of the
                        # same word
                        if getting_repeats:
                            out_sent[replacements[i]] = ''
                        # otherwise we picked a new word to fill <mask> with!
                        else:
                            out_sent[replacements[i]] = word_list[found_ix]
                orig_sent = ' '.join(out_sent)
            # splitting on the last word of the context_sent_throw because we only want to return the sentence beginning
            sent_beginning = orig_sent.split(last_word)[1]
            output.append(sent_beginning)

    # Determining the best sentence beginning
    best_sent = None
    best_score = None
    for i, sent_beginning in enumerate(output):
        score = lm_scoring(sent_beginning)
        word_appears_once1 = sum([w.lower() == word1 for w in sent_beginning.split()]) == 1
        word_appears_once2 = sum([w.lower() == word2 for w in sent_beginning.split()]) == 1
        # first pick a sentence beginning where word1 and word2 only appear once in the sentence beginning
        if best_score is None:
            if word_appears_once1 and word_appears_once2 and isSentenceBeginning(sent_beginning):
                best_sent = sent_beginning
                best_score = score
        # next pick a sentence beginning that has a better score, and word1 and word2 only appear once
        # and there are not punctuation marks
        elif word_appears_once1 and word_appears_once2 and score > best_score and all([punct not in sent_beginning for punct in sentence_ending]) and isSentenceBeginning(sent_beginning):
            best_sent = sent_beginning
            best_score = score
    # if we did not find such a score, lift restrictions and just find the best score using lm_scoring
    if best_sent is None:
        best_sent = output[0]
        best_score = lm_scoring(output[0])
        for i, sent_beginning in enumerate(output):
            score = lm_scoring(sent_beginning)
            if score > best_score and all([punct not in sent_beginning for punct in sentence_ending]) and isSentenceBeginning(sent_beginning):
                best_sent = sent_beginning
                best_score = score
    return best_sent


# Method that uses XLNet to fill in a sentence that has the <mask> tokens already inserted into the middle.
# orig_sent: string the original sentence with <mask> characters already inserted into the string.
# topk: integer topK used for XLNet output
# fill_backwards: boolean, used to fill in XLNet backwards, starting from the last <mask> instead of the first one.
def xl_net_fill_middle(orig_sent, topk=10, fill_backwards=False):
    # setting up XLNet
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
    masked_out = []
    # Filling in the <masks>
    while '<mask>' in orig_sent:
        replacements = [ix for ix, word in enumerate(orig_sent.split()) if word == '<mask>']
        input_ids = torch.tensor(tokenizer.encode(orig_sent, add_special_tokens=True)).unsqueeze(0)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        masked = (input_ids == 6).float()
        perm_mask = perm_mask + masked
        predicts = torch.nonzero(masked[0]).tolist()
        target_mapping = torch.zeros((1, len(predicts), input_ids.shape[1]), dtype=torch.float)
        out_sent = orig_sent.split()
        for n, p in enumerate(predicts):
            target_mapping[0][n][p] = 1.0
        # Running XLNet
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        next_token_logits = outputs[
            0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        # if fill_backwards true fill in the last mask first
        if fill_backwards:
            ranger = list(range(len(predicts) - 1, -1, -1))
        # otherwise fill first <mask> first
        else:
            ranger = list(range(len(predicts)))

        # note we only get one item at a time. Every run of XLNet we will fill in only one <mask>
        for i in ranger[0:1]:
            vals, idxs = torch.topk(next_token_logits[0][i], topk)
            word_list = [tokenizer.decode(idx) for idx in idxs.tolist()]
            found = False
            found_ix = 0
            # Trying to fill in the current <mask> by iterating over the topk choices until we find one that means
            # our specifications
            while not found and found_ix < topk - 1:
                # pw == possible word
                pw = word_list[found_ix]
                # get the word before the <mask> we are trying to fill
                word_before = special_token
                if replacements[i] - 1 > 0 and out_sent[replacements[i] - 1] != '<mask>':
                    word_before = out_sent[replacements[i] - 1]

                # get the word after the <mask> we are trying to fill
                word_after = special_token
                if replacements[i] + 1 < len(out_sent) and out_sent[replacements[i] + 1] != '<mask>':
                    word_after = out_sent[replacements[i] + 1]

                # if the word before it is the end of the sentence and the possible word (pw) is not a punctuation mark
                # and the pw is not a bad start and the pw is not the same as the word before and the word after
                if any([punct in word_before for punct in sentence_ending]) and pw not in sentence_ending \
                        and pw not in avoid_first \
                        and pw not in exclude_words and word_before.lower() != pw.lower() and word_after.lower() != pw.lower():
                    found = True
                # if the word after it is the end of the sentence and the possible word (pw) is not a punctuation mark
                # and the pw is not a bad ending and the pw is not the same as the word before and the word after
                elif any([punct in word_after for punct in sentence_ending]) and pw not in sentence_ending \
                        and pw not in avoid_last \
                        and pw not in exclude_words and word_before.lower() != pw.lower() and word_after.lower() != pw.lower():
                    found = True
                # We are dealing with a middle word
                # pw just has to not be one of the excluded word and not the same as the word before and the word after
                elif not any([punct in word_before for punct in sentence_ending]) and not any(
                        [punct in word_after for punct in sentence_ending]):
                    if pw not in exclude_words and word_before.lower() != pw.lower() and word_after.lower() != pw.lower():
                        found = True
                # otherwise we have not found a good possible word and we move on to the next in topk
                else:
                    found_ix += 1
            if not found:
                print('did not find a sutible word in topk {}'.format(word_list))
                print('Chosing the most top1 word: {}'.format(word_list[0]))
                out_sent[replacements[i]] = word_list[0]
                masked_out.append(word_list[0])
            else:
                out_sent[replacements[i]] = word_list[found_ix]
                masked_out.append(word_list[found_ix])
        orig_sent = ' '.join(out_sent)
    masked_out = ' '.join(masked_out)
    return orig_sent, masked_out


# Method that uses LM Scoring with vanilla GPT-2 to determine the likelihood of a sentence
# Currently we are using the Geometric Mean to make this determination but other methods could be tested TODO
# sentence: string
def lm_scoring(sentence):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)
    score = scorer.sentence_score(sentence, reduce="gmean")
    return score


# Method that uses XLNet to create a sentence beginninng using the one inputted word
# Note that this method is similar to xl_net_fill_begining_two_words
# context_sent_throw: string a sentence that was generated from the desired context that is used to make sure
#                     XLNet starts with the desired context.
# word: string word you want to appear in the generated sentence beginning
# start: integer the minimum number of <masks>
# end: integer the max number of <masks>
# k: integer topK used for XLNet output
# avoid: a list of words you want to avoid generating in the sentence beginning
def xl_net_fill_begining(context_sent_throw, word, start=1, end=6, k=5, avoid=None):
    # setting up XLNet
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
    output = []
    last_word = context_sent_throw.split()[-1]
    # Iterating over different number of <mask>
    for num_masks in range(start, end):
        orig_sent = context_sent_throw + ' <mask> ' * num_masks + word + ' <mask>' * num_masks
        # filling in all the <mask>
        while '<mask>' in orig_sent:
            replacements = [ix for ix, word in enumerate(orig_sent.split()) if word == '<mask>']
            input_ids = torch.tensor(tokenizer.encode(orig_sent, add_special_tokens=True)).unsqueeze(0)
            perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
            masked = (input_ids == 6).float()
            perm_mask = perm_mask + masked
            predicts = torch.nonzero(masked[0]).tolist()
            target_mapping = torch.zeros((1, len(predicts), input_ids.shape[1]), dtype=torch.float)
            out_sent = orig_sent.split()
            for n, p in enumerate(predicts):
                target_mapping[0][n][p] = 1.0
            # running XLNet
            outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
            next_token_logits = outputs[
                0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
            # only fill forward because we are filling a sentence beginning.
            ranger = list(range(len(predicts)))
            # Here we will only look at one <mask> result at a time. This means that we will re-run XLNet after
            # we fill in the current <mask>
            for i in ranger[0:1]:
                vals, idxs = torch.topk(next_token_logits[0][i], k)
                word_list = [tokenizer.decode(idx) for idx in idxs.tolist()]
                found = False
                found_ix = 0
                # Iterating over the topk possible words to find the best one
                while not found and found_ix < k - 1:
                    # pw == possible word
                    pw = word_list[found_ix]
                    # if the possible word is not one we specifically want to avoid and the previous word is not the
                    # same as the possible word, we have found a good word
                    if pw not in avoid and replacements[i] - 1 > 0 and replacements[i] - 1 < len(out_sent) and out_sent[
                        replacements[i] - 1].lower() != pw.lower():
                        found = True
                    # if this is the first word
                    elif replacements[i] == 0 and pw not in avoid:
                        found = True
                    # if this is the last word and it is not a punctuation mark (we want to create the first part
                    # of a sentence so we don't want a short sentence)
                    elif replacements[i] == len(out_sent) - 1 and pw not in sentence_ending:
                        found = True
                    # otherwise, continue looking for another possible word in the topk
                    else:
                        found_ix += 1
                out_sent[replacements[i]] = word_list[found_ix]
            orig_sent = ' '.join(out_sent)
        sentence_beginning = orig_sent.split(last_word)[1]
        output.append(sentence_beginning)

    # find the best sentence beginnig that was generated
    best_sent = None
    best_score = None
    for i, sentence_beginning in enumerate(output):
        score = lm_scoring(sentence_beginning)
        word_appears_once = sum([w.lower() == word for w in sentence_beginning.split()]) == 1
        if best_score is None:
            # if this is the first sentence where the word appears only once choose it to be the best
            if word_appears_once and isSentenceBeginning(sentence_beginning):
                best_sent = sentence_beginning
                best_score = score
        # the word appears only once and we have a better score and it does not have any puntuation
        elif word_appears_once and score > best_score and all([punct not in sentence_beginning for punct in sentence_ending]) and isSentenceBeginning(sentence_beginning):
            best_sent = sentence_beginning
            best_score = score
    # if the best_sent is still None, we relax some restrictions
    if best_sent is None:
        best_sent = output[0]
        best_score = lm_scoring(output[0])
        for i, sentence_beginning in enumerate(output):
            score = lm_scoring(sentence_beginning)
            # now we only care if there is no punctuation and that it is a valid sentence beginning
            if score > best_score and all([punct not in sentence_beginning for punct in sentence_ending]) and isSentenceBeginning(sentence_beginning):
                best_sent = sentence_beginning
                best_score = score
    return best_sent


# Main for Idea 1
def main_idea1():
    pairs = [('gifted2', 'gift_ideas2'), ('strength_training2',
                                          'cookingforbeginners2')]  # , ('scifi','cornell_supreme'),('gift_ideas2','gifted2'), ('strength_training2','cookingforbeginners2'), ('cookingforbeginners2','strength_training2'), ('dnd_bios2', 'kdrama_finetune')]
    for pair in pairs:
        for run in range(2):
            out = idea1(pair[0], pair[1], run, fill_backwards1=False, fill_backwards2=False, use_gpt2=False)


# Main for idea 2
def main_idea2():
    pairs =[('dnd_bios2', 'gift_ideas2', 'elf', 'elf', 'Santa')]
    #pairs = [('strength_training2', 'cookingforbeginners2', 'beat', 'beat', 'eggs'), ('cornell_supreme', 'scifi', 'alien', 'alien', 'spaceship')]
    #pairs = [('strength_training2', 'cookingforbeginners2', 'roll', 'roll', 'butter'),('strength_training2', 'cookingforbeginners2', 'lift', 'lift', 'cupcake'), ('gifted2', 'gift_ideas2', 'gifted', 'gifted', 'child')]#, ('dnd_bios2', 'gift_ideas2', 'elf', 'elf', 'bow'), ('strength_training2', 'cookingforbeginners2', 'twist', 'twist', 'candy'), ('strength_training2', 'cookingforbeginners2', 'roll', 'roll', 'butter'), ('strength_training2', 'cookingforbeginners2', 'beat', 'beat', 'eggs'), ('strength_training2', 'cookingforbeginners2', 'press', 'press', 'tomato')]
    for pair in pairs:
        for run in range(3):
           for length in [30, 60]:
               out = idea2(pair[0], pair[1], word=pair[2], related_word=pair[3], word2=pair[4], run_cnt=run, sentence_len=length, two_words=True, remove_middle=True)
               print('\n\n\n' + out)

# Main for idea 2 Modified
def main_idea2_modified():
    #pairs = [('strength_training2', 'cookingforbeginners2', 'lifted'), ('strength_training2', 'cookingforbeginners2', 'stack'), ('gifted2', 'gift_ideas2', 'gifted'), ('dnd_bios2', 'gift_ideas2', 'elf'), ('dnd_bios2', 'gift_ideas2', 'bow'), ('strength_training2', 'cookingforbeginners2', 'leg')]
    #pairs = [('gifted2', 'gift_ideas2', 'gifted')]
    pairs = [('strength_training2', 'cookingforbeginners2', 'twist'), ('strength_training2', 'cookingforbeginners2', 'roll'), ('strength_training2', 'cookingforbeginners2', 'beat'), ('strength_training2', 'cookingforbeginners2', 'press')]
    for pair in pairs:
        for run in range(4):
            out = idea2_modified(pair[0], pair[1], pair[2], sentence_len=30, run_cnt=run, use_leading=False,
                                 use_combo=False)
            print('\n\n\n' + out)


if __name__ == '__main__':
    main_idea2()
