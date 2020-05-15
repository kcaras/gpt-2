#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
from nltk import sent_tokenize

from util import remove_min_max
import model, sample, encoder, create_graphs
from generate_unconditional_samples import print_logits_of_example

punctuation = [',', '-', '.', '!', ';', '?']
def interact_model(
    model_name='117M',
    run_name='urban_dictionary',
    seed=None,
    nsamples=4,
    batch_size=1,
    length=60,
    temperature=1,
    top_k=40,
    top_p=0.0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p,
            run_name=run_name
        )

        saver = tf.train.Saver([v for v in tf.all_variables() if run_name in v.name])
        ckpt = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


def print_combined_sentences(
    model_name='117M',
    run_name1='gift_ideas2',
    run_name2='gifted2',
    fun_name='urban_dictionary',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=35,
    temperature=1,
    top_k=40,
    top_k_combined=0.0,
    top_p=0.0,
    weight1=0.4,
    weight2=0.3,
    use_random=False,
    use_swap=True,
    logits_used=0,
    ex_num='combined_swap_funny',
    display_logits=True,
    repeat=4,
    use_diverge=False,
    converge_after=1,
    use_funny=True,
):
    """
    Run the sample_model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))
    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    # def score(sentence):
    #     tensor_input = enc.encode(sentence)
    #     loss = model(tensor_input, lm_labels=tensor_input)
    #     return -loss[0] * len(tokenize_input)
    raw_text = ''
    log_dir = ''
    losses0 = []
    losses1 = []
    losses2 = []
    logits_dict = {}
    all_text = []
    oa1 = 0.0
    oa2 = 0.0

    prev_sent_av1 = 0.0
    prev_sent_av2 = 0.0
    converge_count = converge_after
    d = False
    cnt = 0
    while cnt < repeat:
        raw_text = ' '.join(all_text).replace('\n', '').replace('<|endoftext|>', '')
        if use_diverge:
            if cnt > 3:
                d=False
                logits_used=1
            else:
                d=True

        with tf.Session(graph=tf.Graph()) as sess:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            context = tf.placeholder(tf.int32, [batch_size, None])
            ov1 = tf.placeholder(tf.float32)
            ov2 = tf.placeholder(tf.float32)
            output = sample.return_combined_logits(
                hparams=hparams, run_name1=run_name1, run_name2=run_name2,
                length=length + length*cnt,
                start_token=enc.encoder['<|endoftext|>'],
                batch_size=batch_size,
                temperature=temperature,
                top_k=top_k,
                top_k_combined=top_k_combined,
                top_p=top_p,
                weight1=weight1,
                weight2=weight2,
                use_random=use_random,
                use_swap=use_swap,
                logits_used=logits_used,
                display_logits=display_logits,
                diverge=d,
                ov1=ov1,
                ov2=ov2,
                use_funny=use_funny,
                funny_name=fun_name
            )

            saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name1 in v.name])
            ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name1))
            saver1.restore(sess, ckpt1)
            saver2 = tf.train.Saver([v for v in tf.all_variables() if run_name2 in v.name])
            ckpt2 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name2))
            saver2.restore(sess, ckpt2)
            if use_funny:
                saverf = tf.train.Saver([v for v in tf.all_variables() if fun_name in v.name])
                ckptf = tf.train.latest_checkpoint(os.path.join('checkpoint', fun_name))
                saverf.restore(sess, ckptf)
            generated = 0
            while nsamples == 0 or generated < nsamples:
                # feed in sentence from before
                context_tokens = enc.encode(raw_text)
                #print('\nnew: {} len_context: {}\n'.format(raw_text, len(context_tokens)))
                oa1, oa2, out_log, out = sess.run(output,feed_dict={
                        context: [context_tokens for _ in range(batch_size)],
                        ov1: oa1,
                        ov2: oa2
                    })
                print('old av1: {}'.format(oa1))
                print('old av2: {}'.format(oa2))
                out = out[:, len(context_tokens):]
                txt_batch = ''
                for i in range(batch_size):
                    generated += batch_size
                    text = enc.decode(out[i])
                    txt_batch += text
                    #all_text.append(text)
                    sample_str = '\n' + "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n'
                    print(sample_str)
                    print('<!> {}'.format(text))

                norm_str = clean_string(txt_batch)
                lt_arr = sent_tokenize(norm_str)
                print('Printing Sentences')
                sent_cnt = 0
                while cnt < repeat and sent_cnt <= len(lt_arr) -1:
                    sent = lt_arr[sent_cnt].strip()
                    sent_cnt += 1
                    if sent[-1] in ['.', '?', '!'] and sent[0].capitalize() == sent[0] and sent[0] not in punctuation:
                        logits_dict[cnt] = {}
                        logits_dict[cnt]['logits0'] = []
                        logits_dict[cnt]['logits1'] = []
                        logits_dict[cnt]['logits2'] = []
                        logits_dict[cnt]['nums'] = []
                        nums = enc.encode(sent)
                        print(sent)

                        logits0 = [out_log[max(out_log.keys())]['logits'][0][num] for num in nums]
                        logits1 = [out_log[max(out_log.keys())]['logits1'][0][num] for num in nums]
                        logits2 = [out_log[max(out_log.keys())]['logits2'][0][num] for num in nums]

                        probs0 = remove_min_max(logits0)
                        probs1 = remove_min_max(logits1)
                        probs2 = remove_min_max(logits2)

                        if len(probs0) > 0:
                            loss0 = sum(probs0) / len(probs0)
                            loss1 = sum(probs1) / len(probs1)
                            loss2 = sum(probs2) / len(probs2)

                            # currently we want the first to decrease and second to increase
                            # Then we switch to using the first to get the snap

                            if loss1 <= prev_sent_av1: #and loss2 >= prev_sent_av2:
                                converge_count -= 1
                            else:
                                converge_count = converge_after
                            print('Converge cnt: {}, prev1: {} av1: {} prev2 {} av2: {}'.format(converge_count, prev_sent_av1,
                                                                                                loss1, prev_sent_av2, loss2))
                            # if abs(loss0 - loss1) > 0.1:
                            #     print('loss0: {} loss1: {}'.format(loss0, loss1))

                            prev_sent_av1 = loss1
                            prev_sent_av2 = loss2

                            losses0.append(loss0)
                            losses1.append(loss1)
                            losses2.append(loss2)

                            logits_dict[cnt]['logits0'] = logits0
                            logits_dict[cnt]['logits1'] = logits1
                            logits_dict[cnt]['logits2'] = logits2
                            logits_dict[cnt]['nums']= nums
                            cnt += 1

                            all_text.append(sent)
        sess.close()
    print('Converge Count = {}'.format(converge_count))
    losses_dict = {run_name1:losses1, run_name2:losses2, 'combined':losses0}
    if use_funny:
        text_file = 'logs/graphs/{}_{}_{}_{}_{}_{}_{}.txt'.format(ex_num, repeat,
                                                                                                  fun_name,
                                                                                                  run_name1, run_name2,
                                                                                                  weight1, weight2)
    elif top_k_combined > 0:
        text_file = 'logs/graphs/{}_{}_{}_{}_{}_{}_{}.txt'.format(ex_num, repeat,
                                                                                                  'k_combined',
                                                                                                  run_name1, run_name2,
                                                                                                  weight1, weight2)
    elif use_swap:
        text_file = 'logs/graphs/{}_{}_{}_{}_{}_{}_{}.txt'.format(ex_num, repeat,
                                                                                                  'swap',
                                                                                                  run_name1, run_name2,
                                                                                                  weight1, weight2)
    else:
        text_file = 'logs/graphs/{}_{}_{}_{}_{}_{}_{}.txt'.format(ex_num, repeat, logits_used, run_name1, run_name2, weight1, weight2)

    tfile = open(text_file, 'w', encoding='utf-8')
    all_text = [txt.replace('\n', '').replace('<|endoftext|>', '') for txt in all_text]
    tfile.write('\n'.join(all_text))
    tfile.close()
    create_graphs.create_word_chart_many_sents(model_name, run_name1 , run_name2, ex_num, logits_dict, logits_used, display_combined=True)
    print('{} sents'.format(len(losses0)))
    if use_funny:
        print_logits_of_example(
            model_name='117M',
            run_names=(run_name1, run_name2, fun_name),
            seed=None,
            ex_num=ex_num,
            batch_size=1,
            length=length,
            temperature=1,
            top_k=top_k,
            top_p=0.0,
            use_random=False,
            use_swap=False,
            raw_text=' '.join(all_text)
        )

    else:
        print_logits_of_example(
            model_name='117M',
            run_names=(run_name1, run_name2),
            seed=None,
            ex_num=ex_num,
            batch_size=1,
            length=length,
            temperature=1,
            top_k=top_k,
            top_p=0.0,
            use_random=False,
            use_swap=False,
            raw_text=' '.join(all_text)
        )

def interact_combined_model(
        model_name='117M',
        run_name1='strength_training2',
        run_name2='cookingforbeginners2',
        seed=None,
        nsamples=3,
        batch_size=1,
        length=20,
        temperature=1,
        top_k=40,
        top_p=0.0,
        weight1=0.7,
        weight2=0.3,
        use_random=False,
        use_swap=False,
        use_f1=False
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        # w1 = tf.placeholder(tf.float32)
        # w2 = tf.placeholder(tf.float32)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence_combined(hparams=hparams, run_name1=run_name1, run_name2=run_name2,
            length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            weight1=weight1,
            weight2=weight2,
            use_random=use_random,
            use_swap=use_swap,
            use_f1=use_f1)

        saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name1 in v.name])
        ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name1))
        saver1.restore(sess, ckpt1)
        saver2 = tf.train.Saver([v for v in tf.all_variables() if run_name2 in v.name])
        ckpt2 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name2))
        saver2.restore(sess, ckpt2)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                #print('Prompt should not be empty!')
                #raw_text = input("Model prompt >>> ")
                #raw_text = 'How is your day today? Someone commited murder...'
                #raw_text = 'Love is a crazy thing, Your Honor. I killed my stupid, ugly, deadbeat husband so that we could be together. I love you, Your Honor! I hate you!'
                raw_text = 'Love is a crazy thing, Your Honor. I killed my stupid, ugly, deadbeat husband so that we could be together. I love you, Your Honor! You who has such lovely golden locks. Accept my love!!! I plea guilty of the crime that is love!'
                print('Model prompt >>> {}'.format(raw_text))
            # w1 = input('weight for {}'.format(run_name1))
            # if not w1:
            #     w1 = weight1
            # w2 = input('weight for {}'.format(run_name2))
            # if not w2:
            #     w2 = weight2
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                    #w1: tf.convert_to_tensor(w1),
                    #w2: tf.convert_to_tensor(w2)
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


def clean_string(txt):
    punct = ['.', '!', '?']
    for p in punct:
        txt = txt.replace(p, p + ' ')
    return txt


if __name__ == '__main__':
    fire.Fire(interact_model)
