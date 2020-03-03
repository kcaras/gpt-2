from generate_unconditional_samples import sample_combined_models, sample_model
from transformers import XLNetTokenizer, XLNetLMHeadModel
import torch

end_puct = ['.', '!', '?']
# TODO Idea 1
# 1. Get Context Sentence from Domain 1 (GPT-2)
# 2. Find "pivot sentence" that is plausible from both domain 1 and domain 2 (maybe a 50/50 mix from both domains)
# 3. Use XLNet to fill in the sentence with different masks to go between domain 1 and domain 2
# 4. Generate text after pivot using domain 2 (GPT-2)
# Note we can potentially not show the original domain sentence to the end user
def idea1(context1, context2):
    # 1. Get Context Sentence from Domain 1 (GPT-2)
    context_sent1 = 'John and Sue walked together on the beach'
    context_sent1 = sample_model(
        model_name='117M',
        run_name= context1,
        seed=None,
        nsamples=1,
        batch_size=1,
        length=30,
        temperature=1,
        top_k=40,
        top_p=0.0)
    # 2. Find "pivot sentence" that is plausible from both domain 1 and domain 2 (maybe a 50/50 mix from both domains)
    pivot_sentence = 'John got down on one knee'
    pivot_sentence = sample_combined_models(
        model_name='117M',
        run_name1=context1,
        run_name2=context2,
        seed=None,
        nsamples=1,
        batch_size=1,
        length=30,
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
        debug=False)
    # 3. Generate text after pivot using domain 2 (GPT-2)
    context_sent2 = sample_model(
        model_name='117M',
        run_name=context2,
        seed=None,
        nsamples=1,
        batch_size=1,
        length=30,
        temperature=1,
        top_k=40,
        top_p=0.0)
    # 4. Use XLNet to fill in the sentence with different masks to go between domain 1 and domain 2
    orig_sent = context_sent1 + ' <mask> '*4 + context_sent2
    print('context 1: {}'.format(context_sent1))
    print('context 2: {}'.format(context_sent2))
    print('orig_sent: {}'.format(orig_sent))
    out = runXL(orig_sent)
    # Note we can potentially not show the original domain sentence to the end user
    return out


# TODO Idea 2:
# 1. Find a word with multiple senses
# 2. Generate sentence from domain 1 (GPT-2) that uses that word
# 3. Find another word from domain 2 that is related to the word (maybe using word vectors???)
# 4. Generate sentence using that related word in domain 2.
def idea2():
    # 1. Find a word with multiple senses
    # 2. Generate sentence from domain 1 (GPT-2) that uses that word
    # 3. Find another word from domain 2 that is related to the word (maybe using word vectors???)
    # 4. Generate sentence using that related word in domain 2.

    pass


def runXL(orig_sent):
    # getting the model
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
    model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

    # prepping the data
    seen_words = []
    # orig_sent = '<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> John gets down on one knee to Sarah\'s surprise. John ties his shoe.'
    prev_word = ''
    outf = open('out.txt', 'w', encoding='utf-8')
    max_cnt = sum([1 for word in orig_sent.split() if word == '<mask>']) * 3
    outf.write(orig_sent + '\n')
    while '<mask>' in orig_sent:
        max_cnt -= 1
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
        for i in range(len(predicts)):
            print("mask", i)
            outf.write("mask {}\n".format(i))
            if max_cnt > 0:
                vals, idxs = torch.topk(next_token_logits[0][i], 5)
            else:
                vals, idxs = torch.topk(next_token_logits[0][i], 5)
            print(vals, idxs)
            print(idxs.tolist())
            for idx in idxs.tolist():
                new_word = tokenizer.decode(idx)
                if new_word not in seen_words and max_cnt > 0:
                    # if prev_word == '' or new_word != prev_word:
                    outf.write('new: {}\n'.format(new_word))
                    out_sentence[replacements[replace]] = new_word
                    seen_words.append(new_word)
                    prev_word = new_word
                elif max_cnt > 0:
                    print('Already Seen: {}'.format(new_word))
                    outf.write('Already Seen: {}\n'.format(new_word))
                else:
                    print('max_cnt exceeded, just filling in the sentence')
                    outf.write('max_cnt exceeded, just filling in the sentence\n')
                    outf.write('filling: {}'.format(new_word))
                    out_sentence[replacements[replace]] = new_word
                    prev_word = new_word
            replace += 1
        orig_sent = ' '.join(out_sentence)
        print(orig_sent)
        outf.write(orig_sent + '\n')
    outf.close()
    return orig_sent


if __name__ == '__main__':
    out = idea1('gifted2', 'gift_ideas2')
    print(out)