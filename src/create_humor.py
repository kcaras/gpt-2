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
exclude_words = ['<eop>', '<eod>']
avoid_xlnet_idea2 = ['.', ',', '?', '!', '<eop>', 'and', 'but', '<eop>', '<eod>']
special_token = '<**>'
avoid_first = ['and', 'but']
avoid_last = ['and', 'but', 'so']


# TODO Idea 1
# 1. Get Context Sentence from Domain 1 (GPT-2)
# 2. Find "pivot sentence" that is plausible from both domain 1 and domain 2 (maybe a 50/50 mix from both domains)
# 3. Use XLNet to fill in the sentence with different masks to go between domain 1 and domain 2
# 4. Generate text after pivot using domain 2 (GPT-2)
# Note we can potentially not show the original domain sentence to the end user
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
    #client = GrammarBotClient()
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
        # print(pivot_sentence)
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
        # context_sent2 = sent_tokenize(context_sent2)[0]
        # print(context_sent2)
        if len(sent_tokenize(context_sent2)) >= 2:
            context_sent2 = ' '.join(sent_tokenize(context_sent2)[0:2])
        else:
            context_sent2 = sent_tokenize(context_sent2)[0]
        isSent = isSentence(context_sent2)
    context_sent2 = context_sent2.replace('\n', '')
    # 4. Use XLNet to fill in the sentence with different masks to go between domain 1 and domain 2
    # TODO Try different mask lengths and choose the "best" one
    print('context 1: {}'.format(context_sent1))
    print('context 2: {}'.format(context_sent2))
    best_sent = ''
    best_val = 100
    best_num_masks = 0
    start = 4
    end = 20
    best_before = ''
    context2_words = context_sent2.split()
    split_len = len(context_sent2.split()) // 2
    for i in range(start, end):
        if use_gpt2:
            orig_sent1 = context_sent1 + pivot_sentence + ' '
            is_good = False
            while not is_good:
                first_half = sample_model_with_seed(model_name='117M', run_name='model', seed=None, nsamples=1,
                                                    batch_size=1, length=i, temperature=1, top_k=40, top_p=0.0,
                                                    raw_text=orig_sent1)
                is_good = 'www.' not in first_half
            orig_sent1 = orig_sent1 + first_half + ' <mask> ' * (i // 2) + context_sent2
            print('first part: {}'.format(orig_sent1))
            f.write('\nfirst part: {}\n'.format(first_half))
            f.write('orig_sent:{}\n'.format(orig_sent1))
            before = orig_sent1
            out_sent, masked_out1 = xl_net_fill_middle(orig_sent1, fill_backwards=fill_backwards1)
            print('\nout_sent: {}\n'.format(out_sent))
            f.write('out_sent: {}\n'.format(out_sent))
            masked_part = first_half + masked_out1
            score = score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_part, context_sent2)

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
            # res = client.check(masked_out2)
            # score = -1.0*len(res.matches)*0.3 + i*0.7
            score = score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_out2, context_sent2)
            # f.write('score\n\n: {}'.format(len(res.matches)))
        else:
            orig_sent1 = context_sent1 + pivot_sentence + ' <mask> ' * i + context_sent2
            before = orig_sent1
            print('{} masks'.format(i))
            f.write('\n{} masks\n'.format(i))
            f.write('orig: {}'.format(orig_sent1))
            out_sent, masked_out = xl_net_fill_middle(orig_sent1, fill_backwards=fill_backwards1)
            print('\nout_sent2: {}\n'.format(out_sent))
            f.write('out_sent2: {}\n'.format(out_sent))
            score = score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_out, context_sent2)

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
    # Note we can potentially not show the original domain sentence to the end user
    return best_sent


def score_sentence1(context_sent1, pivot_sentence, masked_sentence, context_sent2):
    client = GrammarBotClient()
    sentence = context_sent1 + pivot_sentence + masked_sentence + context_sent2
    res = client.check(sentence)
    score = len(res.matches)
    return score


def score_grammar(sentence):
    client = GrammarBotClient()
    res = client.check(sentence)
    score = len(res.matches)
    return score

def score_sentence_lm_scoring(context_sent1, pivot_sentence, masked_sentence, context_sent2):
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
    grammar_score = sum([lm_scoring(text) for text in [context_sent1, pivot_sentence, masked_sentence, context_sent2]])/4.0
    #grammar_score = score_sentence1(context_sent1, pivot_sentence, masked_sentence, context_sent2) / len_ex
    score = (sentiment_difference1 + sentiment_difference2 - grammar_score) / 4 
    return score


def score_sentence2(context_sent1, pivot_sentence, masked_sentence, context_sent2):
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
    grammar_score = score_sentence1(context_sent1, pivot_sentence, masked_sentence, context_sent2) / len_ex
    score = (sentiment_difference1 + sentiment_difference2 - grammar_score) / 4
    return score


def isSentence(sentence, at_least_length=10):
    has_word = len(sentence.split()) > 0
    sentence = sentence.strip()
    return has_word and sentence != '' and any([sentence[-1] == punct for punct in sentence_ending]) and len(
        sentence.split()) > at_least_length and 'www.' not in sentence and 'http' not in sentence and '/r' not in sentence


def isSentenceBeginning(sb, at_least_length=3):
    sb = sb.strip()
    return all([sb[-1] != punct for punct in sentence_ending]) and len(sb.split()) > at_least_length

# TODO Idea 2:
# 1. Find a word with multiple senses (word)
# 2. Generate sentence from domain 1 (GPT-2) that uses that word
# 3. Find another word from domain 2 that is related to the word (related_word)
# 4. Generate sentence using that related word.
def idea2(context1, context2, word='', related_word='', word2='', run_cnt=0, sentence_len=30):
    # 1. Find a word with multiple senses
    if word == '':
        word = 'gifted'
    else:
        print('word: {}'.format(word))
    # Generate a throw away word to start out the XLNet
    isSent = False
    f = open('idea2_samples/lm_scoring/idea2_{}_len_{}_{}_{}_{}_{}_{}.txt'.format(run_cnt, sentence_len, context1, context2, word, related_word, word2),
             'w', encoding='utf-8')
    f.write('word: {}\nrelated_word: {}\nword2:{}\n'.format(word, related_word, word2))
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

    print('context_sent1_throw: {}'.format(context_sent1_throw))
    f.write('context_sent1_throw: {}\n'.format(context_sent1_throw))
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    seed_text1 = xl_net_fill_begining(context_sent1_throw, word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    # seed_text1 = 'He {} '.format(word)
    print('seed_text1: {}'.format(seed_text1))
    f.write('\nseed_text1: {}\n'.format(seed_text1))
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
                                               raw_text=seed_text1)
        context_sent1 = sent_tokenize(seed_text1 + context_sent1)
        context_sent1 = ' '.join(context_sent1[:-1])
        isSent = isSentence(context_sent1)
    print('context_sent1: {}'.format(context_sent1))
    f.write('\ncontext_sent1: {}\n'.format(context_sent1))
    # 3. Find another word from domain 2 that is related to the word (maybe using word vectors or frequencies???)
    if related_word == '':
        related_word = 'child'
    else:
        print('related_word: {}'.format(related_word))
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
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    seed_text2 = xl_net_fill_begining(context_sent2_throw, related_word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    print('seed_text2: {}'.format(seed_text2))
    # 4. Generate sentence using that related word in domain 2.
    # seed_text2 = 'The {} '.format(related_word)
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
                                               raw_text=context_sent1 + ' ' + seed_text2
                                               )
        context_sent2 = sent_tokenize(seed_text2 + context_sent2)
        context_sent2 = ' '.join(context_sent2[:-1])
        isSent = isSentence(context_sent2)
    print('context_sent2: {}'.format(context_sent2))
    # 5. Find word only used in domain 2
    if word2 == '':
        word2 = 'pancake'
    else:
        print('word2 {}'.format(word2))
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
    seed_text3 = xl_net_fill_begining(context_sent2_throw2, word2, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    print('seed_text3: {}'.format(seed_text3))
    # seed_text3 = 'This {}'.format(word2)
    isSent = False
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
                                               raw_text=context_sent1 + ' ' + context_sent2 + ' ' + seed_text3)
        context_sent3 = sent_tokenize(seed_text3 + context_sent3)
        context_sent3 = ' '.join(context_sent3[:-1])
        isSent = isSentence(context_sent3)
    print('context_sent3: {}'.format(context_sent3))
    print('\n\n')
    print('seed_text1: {}'.format(seed_text1))
    print('context_sent1: {}'.format(context_sent1))
    print('seed_text2: {}'.format(seed_text2))
    print('context_sent2: {}'.format(context_sent2))
    print('seed_text3: {}'.format(seed_text3))
    print('context_sent3: {}'.format(context_sent3))
    print('context_sent3: {}'.format(context_sent3))
    f.write('\n\n')
    f.write('seed_text1: {}\n'.format(seed_text1))
    f.write('context_sent1: {}\n'.format(context_sent1))
    f.write('seed_text2: {}\n'.format(seed_text2))
    f.write('context_sent2: {}\n'.format(context_sent2))
    f.write('seed_text3: {}\n'.format(seed_text3))
    f.write('context_sent3: {}\n'.format(context_sent3))
    f.write('\n\n\n')
    # if remove_middle:
    #    f.write(context_sent1 + ' ' + context_sent3)
    #    f.close()
    #    return context_sent1 + ' ' + context_sent3
    # else:
    f.write(context_sent1 + ' ' + context_sent2 + ' ' + context_sent3)
    f.close()
    return context_sent1 + ' ' + context_sent2 + ' ' + context_sent3


# 1. Find a word with multiple senses (word)
# 2. Generate sentence from domain 1 (GPT-2) that uses that word
# 3a. Could possibly generate lead sentence in domain 2
# 3. Generate sentence from domain 2 (GPT-2) that uses that word
# 4. Generate sentence using that related word.
def idea2_modified(context1, context2, word='', run_cnt=0, sentence_len=40, use_leading=True, use_combo=False):
    # 1. Find a word with multiple senses
    if word == '':
        word = 'gifted'
    else:
        print('word: {}'.format(word))
    # Generate a throw away word to start out the XLNet
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
    # Used to generate a leading sentence for XLNet
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
    print('context_sent1_throw: {}'.format(context_sent1_throw))
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    seed_text1 = xl_net_fill_begining(context_sent1_throw, word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    # seed_text1 = 'He {} '.format(word)
    print('seed_text1: {}'.format(seed_text1))
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
                                               raw_text=seed_text1)
        context_sent1 = sent_tokenize(seed_text1 + context_sent1)
        context_sent1 = ' '.join(context_sent1[:-1])
        isSent = isSentence(context_sent1)
    print('context_sent1: {}'.format(context_sent1))
    # 3. Generate sentence from domain 2 (GPT-2) that uses that word
    # context_sent2_throw generated to help XLNet
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
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    seed_text2 = xl_net_fill_begining(context_sent2_throw, word, start=1, end=6, k=5, avoid=avoid_xlnet_idea2)
    print('seed_text2: {}'.format(seed_text2))
    context2_input = ''
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
        context2_input = context_sent1 + ' ' + context_sent2_throw2 + ' ' + seed_text2
        # seed_text3 = 'This {}'.format(word2)
    else:
        context2_input = context_sent1 + ' ' + seed_text2

    # 3. Generate sentence using that word in domain 2.
    isSent = False
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
            context_sent2 = sent_tokenize(seed_text2 + context_sent2)
            context_sent2 = ' '.join(context_sent2[:-1])
            isSent = isSentence(context_sent2)
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
            context_sent2 = sent_tokenize(seed_text2 + context_sent2)
            context_sent2 = ' '.join(context_sent2[:-1])
            isSent = isSentence(context_sent2)

    print('context_sent2: {}'.format(context_sent2))
    print('\n\n')
    print('seed_text1: {}'.format(seed_text1))
    print('context_sent1: {}'.format(context_sent1))
    print('seed_text2: {}'.format(seed_text2))
    print('context_sent2: {}'.format(context_sent2))
    if use_leading:
        print('context2_sent_throw2:  {}'.format(context_sent2_throw2))
        f.write('context2_sent_throw2:  {}'.format(context_sent2_throw2))
    f.write('\n\n')
    f.write('seed_text1: {}\n'.format(seed_text1))
    f.write('context_sent1: {}\n'.format(context_sent1))
    f.write('seed_text2: {}\n'.format(seed_text2))
    f.write('context_sent2: {}\n'.format(context_sent2))
    f.write('\n\n\n')
    f.write(context_sent1 + ' ' + context_sent2)
    f.close()
    return context_sent1 + ' ' + context_sent2


def get_sentiment(sentence):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    return sentiment_analyzer.polarity_scores(sentence)['compound']


def runXL_old(orig_sent, fill_backwards=False, topk=5):
    # getting the model
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

    # prepping the data
    seen_words = []
    # orig_sent = '<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> John gets down on one knee to Sarah\'s surprise. John ties his shoe.'
    prev_word = ''
    outf = open('out.txt', 'w', encoding='utf-8')
    masked_idx = [ix for ix, word in enumerate(orig_sent.split()) if word == '<mask>']
    max_cnt = sum([1 for word in orig_sent.split() if word == '<mask>']) * 10
    outf.write(orig_sent + '\n')
    while '<mask>' in orig_sent:
        # max_cnt -= 1
        input_ids = torch.tensor(tokenizer.encode(orig_sent, add_special_tokens=True)).unsqueeze(0)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        masked = input_ids == 6
        perm_mask = perm_mask + masked
        predicts = torch.nonzero(masked[0]).tolist()
        target_mapping = torch.zeros((1, len(predicts), input_ids.shape[1]), dtype=torch.float)
        for n, p in enumerate(predicts):
            target_mapping[0][n][p] = 1.0

        # Run the model
        outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
        next_token_logits = outputs[
            0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
        out_sentence = orig_sent.split()
        first_mask_ix = out_sentence.index('<mask>')

        # filling the masks in
        replacements = [ix for ix, word in enumerate(out_sentence) if word == '<mask>']
        replace = 0
        if fill_backwards:
            ranger = list(range(len(predicts)))
        else:
            ranger = list(range(len(predicts) - 1, -1, -1))
        for i in ranger:
            # print("mask", i)
            outf.write("mask {}\n".format(i))
            if max_cnt > 0:
                vals, idxs = torch.topk(next_token_logits[0][i], topk)
            else:
                vals, idxs = torch.topk(next_token_logits[0][i], topk)
            # print(vals, idxs)
            # print(idxs.tolist())
            idxs1 = idxs.tolist()
            # filled_one = False
            # new_words = [tokenizer.decode(idx) for idx in idxs1]
            # print('new_words {}'.format(new_words))
            # if fill_backwards:
            #    ranger2  = range(len(idxs1)-1, -1, -1)
            # else:
            ranger2 = range(len(idxs1))
            print('Possible Words: {}'.format([tokenizer.decode(idxs1[ix]) for ix in ranger2]))
            print('Should be {}: {}'.format(len(list(ranger2)), topk))
            found = False
            ix = 0
            while not found and ix < len(idxs1):
                # for ix in ranger2:
                idx = idxs1[ix]
                new_word = tokenizer.decode(idx)
                # if max_cnt > 0 and new_word not in exclude_words:
                prev_word = out_sentence[replacements[replace] - 1]
                print('potential word: {} previous word: {}'.format(new_word, prev_word))
                if new_word not in exclude_words and new_word != prev_word:
                    outf.write('new: {}\n'.format(new_word))
                    out_sentence[replacements[replace]] = new_word
                    # print('\n***************************************************************\n')
                    # print('cur_sent replacing: {}'.format(out_sentence))
                    # if new_word not in sentence_ending:
                    seen_words.append(new_word)
                    # prev_word = new_word
                    # filled_one = True
                    replace += 1
                    found = True
                elif max_cnt > 0 and new_word not in exclude_words:
                    # print('Already Seen: {}'.format(new_word))
                    outf.write('Already Seen: {}\n'.format(new_word))
                ix += 1
                # elif max_cnt <= 0 and new_word not in exclude_words:
                # print('max_cnt exceeded, just filling in the sentence')
                # outf.write('max_cnt exceeded, just filling in the sentence\n')
                # outf.write('filling: {}'.format(new_word))
                # out_sentence[replacements[replace]] = new_word
                # print(out_sentence)
                # prev_word = new_word
            # replace +=1
        orig_sent = ' '.join(out_sentence)
        print(orig_sent)
        outf.write(orig_sent + '\n')
    outf.close()
    masked_output = ' '.join(out_sentence[masked_idx[0]:masked_idx[-1]])
    return orig_sent, masked_output


def xl_net_fill_middle(orig_sent, topk=10, fill_backwards=False):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
    masked_out = []
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
        if fill_backwards:
            ranger = list(range(len(predicts) - 1, -1, -1))
        else:
            ranger = list(range(len(predicts)))
        for i in ranger[0:1]:
            print("mask", i)
            vals, idxs = torch.topk(next_token_logits[0][i], topk)
            word_list = [tokenizer.decode(idx) for idx in idxs.tolist()]
            # print(vals, idxs)
            # print(idxs.tolist())
            print('Top {}: {}'.format(topk, word_list))
            found = False
            found_ix = 0
            while not found and found_ix < topk - 1:
                pw = word_list[found_ix]
                word_before = special_token
                if replacements[i] - 1 > 0 and out_sent[replacements[i] - 1] != '<mask>':
                    word_before = out_sent[replacements[i] - 1]
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
                else:
                    found_ix += 1
            if not found:
                print('did not find a sutible word in topk {}'.format(word_list))
                print('Chosing the most likely word: {}'.format(word_list[0]))
                out_sent[replacements[i]] = word_list[0]
                masked_out.append(word_list[0])
            else:
                out_sent[replacements[i]] = word_list[found_ix]
                masked_out.append(word_list[found_ix])
        orig_sent = ' '.join(out_sent)
    masked_out = ' '.join(masked_out)
    return orig_sent, masked_out

def lm_scoring(sentence):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)
    score = scorer.sentence_score(sentence, reduce="gmean")
    return score

def xl_net_fill_begining(context_sent_throw, word, start=1, end=6, k=5, avoid=None):
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')
    output = []
    if len(context_sent_throw.split()) < 1:
        print('context_sent_throw: {}'.format(context_sent_throw))
    last_word = context_sent_throw.split()[-1]
    for num_masks in range(start, end):
        orig_sent = context_sent_throw + ' <mask> ' * num_masks + word + ' <mask>' * num_masks
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
            # ranger = list(range(len(predicts)-1, -1, -1))
            ranger = list(range(len(predicts)))
            for i in ranger[0:1]:
                # print("mask", i)
                vals, idxs = torch.topk(next_token_logits[0][i], k)
                word_list = [tokenizer.decode(idx) for idx in idxs.tolist()]
                # print(vals, idxs)
                # print(idxs.tolist())
                # print('Top {}: {}'.format(k, word_list))
                found = False
                found_ix = 0
                while not found and found_ix < k - 1:
                    pw = word_list[found_ix]
                    if pw not in avoid and replacements[i] - 1 > 0 and replacements[i] - 1 < len(out_sent) and out_sent[
                        replacements[i] - 1].lower() != pw.lower():
                        found = True
                    elif replacements[i] == 0 and pw not in avoid:
                        found = True
                    elif replacements[i] == len(out_sent) - 1 and pw not in sentence_ending:
                        found = True
                    else:
                        found_ix += 1
                out_sent[replacements[i]] = word_list[found_ix]
            orig_sent = ' '.join(out_sent)
        first_last = orig_sent.split(last_word)
        output.append(first_last)

    best_sent = None
    best_score = None
    for i, first_last in enumerate(output):
        # print('mask num: {}'.format(start + i))
        # for text in first_last:
        #    print('\nOut Sent: {}\n'.format(text))
        score = lm_scoring(first_last[1])
        word_appears_once = sum([w.lower() == word for w in first_last[1].split()]) == 1
        if best_score is None:
            if word_appears_once and isSentenceBeginning(first_last[1]):
                best_sent = first_last[1]
                best_score = score
        elif word_appears_once and score > best_score and all([punct not in first_last[1] for punct in sentence_ending]) and isSentenceBeginning(first_last[1]):
            best_sent = first_last[1]
            best_score = score
    return best_sent


def main_idea1():
    pairs = [('gifted2', 'gift_ideas2'), ('strength_training2',
                                          'cookingforbeginners2')]  # , ('scifi','cornell_supreme'),('gift_ideas2','gifted2'), ('strength_training2','cookingforbeginners2'), ('cookingforbeginners2','strength_training2'), ('dnd_bios2', 'kdrama_finetune')]
    for pair in pairs:
        for run in range(2):
            out = idea1(pair[0], pair[1], run, fill_backwards1=False, fill_backwards2=False, use_gpt2=False)


def main_idea2():
   pairs = [('strength_training2', 'cookingforbeginners2', 'roll', 'roll', 'butter'),
             ('gifted2', 'gift_ideas2', 'gifted', 'gifted', 'child')]#, ('dnd_bios2', 'gift_ideas2', 'elf', 'elf', 'bow'), ('strength_training2', 'cookingforbeginners2', 'twist', 'twist', 'candy'), ('strength_training2', 'cookingforbeginners2', 'roll', 'roll', 'butter'), ('strength_training2', 'cookingforbeginners2', 'beat', 'beat', 'eggs'), ('strength_training2', 'cookingforbeginners2', 'press', 'press', 'tomato')]
   for pair in pairs:
       for run in range(3):
           out = idea2(pair[0], pair[1], word=pair[2], related_word=pair[3], word2=pair[4], run_cnt=run, sentence_len=30)
           print('\n\n\n' + out)


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
