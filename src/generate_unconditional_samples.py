#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fire
import json
import os
import numpy as np
import tensorflow as tf
from util import remove_min_max

import model, sample, encoder, create_graphs
import json
from nltk import sent_tokenize
romance_sents = 'But youth asked nothing of its parents -- not a touch of the hand or a kiss given in passing.The only thing unusual about the Old Man had long since happened . But the past was dead here as the present was dead . Once the Old Man had had a wife. And once she , too , ignored him . With a tiny fur-piece wrapped around her shoulders , she wiggled her satin-covered buttocks down the street before him and didn\'t stop.'
supreme_sents = 'We will now hear argument in the Cherokee Nation against Thompson and Thompson against the Cherokee Nation. Mr. Miller. Justice Stevens, and may it please the Court: These two contract cases concern whether the Government is liable in money damages under the Contract Disputes Act and section 110 of the Indian Self-Determination Act when the Secretary fails to fully pay a contract price for the -- Would you mind explaining to us how these two cases relate? The Court of Appeals for the Federal Circuit decision went one way and the Tenth Circuit went another. And are the claims at all overlapping? How are they differentiated?'
movie_sents = 'I\'m serious, man, he\'s whacked.  He sold his own liver on the black market so he could buy new speakers. They always let felons sit in on Honors Biology? No kidding.  He\'s a criminal.  I heard he lit a state trooper on fire.  He just got out of Alcatraz... He seems like he thrives on danger. What makes you think he\'ll do it? You wanna go out with him? What about him? Unlikely, but even so, she still can\'t go out with you.  So what\'s the point? I teach her French, get to know her, dazzle her with charm and she falls in love with me.'
scifi_sents = 'Alien envoys come to the giant space station. In the earth year 2257, a multitude of humans and non-humans gather deep in neutral space at a new station, Babylon 5. Some of them are members of the station crew, including Commander Jeffrey Sinclair, Lieutenant Commander Laurel Takashima, Security Chief Michael Garibaldi, and Medical Officer Benjamin Kyle. Others are ambassadors from major alien governments: Ambassador G\'Kar from the Narn Regime, Ambassador Delenn from the Minbari Federation, and Ambassador Londo Mollari from the Centauri Republic. Still others are refugees, smugglers, businessmen, diplomats, and travelers from a hundred worlds.'
court2scifi = 'Calling the court to order we will hear the case of John Doe. Lawyers, what are your opening arguements. Well your honor, my client clearly is an alien. He did park his spaceship illegally. But, he moved it at ludicrous speed!'
court2scifi1 = 'We will now hear argument in the Cherokee Nation against Thompson and Thompson against the Cherokee Nation. Mr. Miller. Well your honor, my client clearly is an alien lifeform. His spaceship was moved at ludicrous speed. Geep Gorp has no problems with the Cherokee. Why would a transflexian hate anyone?'
kdrama_sents = 'At the office, Jae-in mindlessly twirls a pen while scanning a document as his employees sit on pins and needles. They each glance at their boss until someone’s phone breaks the tense silence. The guilty employee turns it off in haste, and Jae-in appears to pardon the disruption. He begins to speak when another cellphone rings, but before the employee can grab his phone, Jae-in intercepts it and tosses it to the floor. Without acknowledging what just happened, Jae-in asks if everyone is now ready for their meeting.'

def sample_model(
    model_name='117M',
    seed=None,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0
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

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)


def sample_combined_models(
    model_name='117M',
    run_name1='cornell_supreme',
    run_name2='kdrama_finetune',
    seed=None,
    nsamples=20,
    batch_size=1,
    length=200,
    temperature=1,
    top_k=40,
    top_k_combined=30,
    top_p=0.0,
    weight1=0.3,
    weight2=0.7,
    use_random=False,
    use_swap=False,
    use_fifty_one=False,
    debug=False
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

    # with tf.variable_scope(run_name1):
    #     en_fr_model = create_model(...)
    # with tf.variable_scope(run_name2):
    #     fr_en_model = create_model(...)
    # tester_body = open('testy_body.txt', 'a', encoding='utf-8')
    # tester_body.write('calling sample_sequence\n')
    # tester_body.close()
    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence_combined(
            hparams=hparams, run_name1=run_name1, run_name2=run_name2,
            length=length,
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
            use_fifty_one=use_fifty_one,
            debug=debug
        )[:, 1:]

        saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name1 in v.name])
        ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name1))
        saver1.restore(sess, ckpt1)
        saver2 = tf.train.Saver([v for v in tf.all_variables() if run_name2 in v.name])
        ckpt2 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name2))
        saver2.restore(sess, ckpt2)
        if top_k_combined > 0:
            out_file = 'samples/{}_{}/two_pasts/static_weights/temp_{}_len_{}_p_{}_combined_k_{}_w1_{}_w2_{}.txt'.format(
                run_name1, run_name2, temperature, str(length), top_p, top_k_combined, weight1, weight2)
        elif use_fifty_one:
            out_file = 'samples/{}_{}/two_pasts/fifty_one_weights/rand_temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1, run_name2, temperature, str(length), top_p, top_k, weight1, weight2)
        elif use_random:
            out_file = 'samples/{}_{}/two_pasts/random_weights/rand_temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1, run_name2, temperature, str(length),
                                                                                top_p, top_k, weight1, weight2)
        elif use_swap:
            out_file = 'samples/{}_{}/two_pasts/swap_weights/swap_temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1,
                                                                                                           run_name2,
                                                                                                           temperature,
                                                                                                           str(length),
                                                                                                           top_p, top_k,
                                                                                                           weight1,
                                                                                                           weight2)
        else:
            out_file = 'samples/{}_{}/two_pasts/static_weights/temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1, run_name2,temperature, str(length), top_p, top_k, weight1, weight2)
        f = open(out_file, 'w', encoding='utf-8')
        #tester = open('testy_gen.txt', 'w', encoding='utf-8')

        #testLines = []
        cnt = 0
        generated = 0
        if debug:
            #merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(
                '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/gpt-2/logs')
            writer.add_graph(sess.graph)
            out = sess.run(output)
            cnt += 1
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                sample_str = '\n' + "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n'
                print(sample_str)
                print(text)
                f.write(sample_str)
                f.write(text)
            f.close()
        else:
            while nsamples == 0 or generated < nsamples:
                # f = open('testy1.txt', 'a', encoding='utf-8')
                # f.write('In generated sample\n')
                # f.close()
                # f = open('testy.txt', 'w', encoding='utf-8')
                # f.write('In generated sample\n')
                # # f.close()
                # tester_body = open('testy_body.txt', 'a', encoding='utf-8')
                # tester_body.write('About to call run\n')
                # tester_body.close()
                out = sess.run(output)
                cnt += 1
                for i in range(batch_size):
                    generated += batch_size
                    text = enc.decode(out[i])
                    sample_str = '\n' + "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n'
                    print(sample_str)
                    print(text)
                    f.write(sample_str)
                    f.write(text)
            # testLines.append('cnt: {}\n'.format(cnt))
            # tester.writelines(testLines)
            f.close()


def print_logits(
    model_name='117M',
    run_name1='brown_romance',
    run_name2='cornell_supreme',
    seed=None,
    nsamples=15,
    batch_size=1,
    length=None,
    temperature=2,
    top_k=40,
    top_p=0.0,
    weight1=0.5,
    weight2=0.5,
    use_random=False,
    use_swap=False
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

    # with tf.variable_scope(run_name1):
    #     en_fr_model = create_model(...)
    # with tf.variable_scope(run_name2):
    #     fr_en_model = create_model(...)

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.return_logits(
            hparams=hparams, run_name1=run_name1, run_name2=run_name2,
            length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            weight1=weight1,
            weight2=weight2,
            use_random=use_random,
            use_swap=use_swap
        )

        saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name1 in v.name])
        ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name1))
        saver1.restore(sess, ckpt1)
        saver2 = tf.train.Saver([v for v in tf.all_variables() if run_name2 in v.name])
        ckpt2 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name2))
        saver2.restore(sess, ckpt2)
        # if use_random:
        #     out_file = 'samples/{}_{}/two_pasts/random_weights/rand_temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1, run_name2, temperature, str(length),
        #                                                                         top_p, top_k, weight1, weight2)
        # elif use_swap:
        #     out_file = 'samples/{}_{}/two_pasts/swap_weights/swap_temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1,
        #                                                                                                    run_name2,
        #                                                                                                    temperature,
        #                                                                                                    str(length),
        #                                                                                                    top_p, top_k,
        #                                                                                                    weight1,
        #                                                                                                    weight2)
        # else:
        #     out_file = 'samples/{}_{}/two_pasts/static_weights/temp_{}_len_{}_p_{}_k_{}_w1_{}_w2_{}.txt'.format(run_name1, run_name2,temperature, str(length), top_p, top_k, weight1, weight2)
        #f = open(out_file, 'w', encoding='utf-8')
        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            print(out)
            # for i in range(batch_size):
            #     generated += batch_size
            #     text = enc.decode(out[i])
            #     sample_str = '\n' + "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n'
            #     f.write(sample_str)
            #     f.write(text)
            #     print(sample_str)
            #     print(text)
        #f.close()


def print_combined_logits(
    model_name='117M',
    run_name1='scifi',
    run_name2='kdrama_finetune',
    seed=None,
    nsamples=1,
    batch_size=20,
    length=40,
    temperature=1,
    top_k_combined=120,
    top_k=40,
    top_p=0.0,
    weight1=0.5,
    weight2=0.5,
    use_random=False,
    use_swap=False,
    use_fifty_one=False,
    debug=True,
    logits_used=0,
    ex_num='ex_topkcombined',
    display_logits=True
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

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.return_combined_logits(
            hparams=hparams, run_name1=run_name1, run_name2=run_name2,
            length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature,
            top_k_combined=top_k_combined,
            top_k=top_k,
            top_p=top_p,
            weight1=weight1,
            weight2=weight2,
            use_random=use_random,
            use_swap=use_swap,
            logits_used=logits_used,
            display_logits=display_logits
        )

        saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name1 in v.name])
        ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name1))
        saver1.restore(sess, ckpt1)
        saver2 = tf.train.Saver([v for v in tf.all_variables() if run_name2 in v.name])
        ckpt2 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name2))
        saver2.restore(sess, ckpt2)
        generated = 0
        names = [run_name1, run_name2, 'combined']
        log_dir = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/logs'
        while nsamples == 0 or generated < nsamples:
            out_log, out = sess.run(output)
            for i in out_log.keys():
                for j, logy in enumerate(['logits1', 'logits2', 'logits']):
                    out_file1 = '{}/{}/{}/{}_{}/{}/{}_{}.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, names[j], i, logy)
                    odict = {}
                    of1 = open(out_file1, 'w', encoding='utf-8')
                    logits = out_log[i][logy]
                    for k in range(logits.shape[1]):
                        val = logits[0][k]
                        sym = enc.decoder[k]
                        #of1.write('{},{}\n'.format(sym, val))
                        odict[str(sym)] = float(val)
                    json.dump(odict, of1)
                    of1.close()
            # out_file = 'logs/{}_{}/logits1/{}'
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                sample_str = '\n' + "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n'
                # f.write(sample_str)
                # f.write(text)
                print(sample_str)
                print(text)
                print(out[i].shape)
                print(out_log[0]['logits1'].shape)
                #print(out_log[0]['logits1'][0])
                #print(enc.decode())
                print(out_log[0]['logits1'][0][0])
                #print(out_log[0]['logits1_idxs'].shape)
                #print(out_log[0]['logits1_idxs'][0])
                text_file = '{}/{}/{}/{}_{}/text.txt'.format(log_dir, ex_num, logits_used, run_name1, run_name2)
                tfile = open(text_file, 'w', encoding='utf-8')
                tfile.write(text.replace('\n', '') + '\n')
                nums = [str(out[i][z]) for z in range(out[i].shape[0])]
                tfile.write(' '.join(nums))
                tfile.close()

        create_graphs.create_word_chart(model_name, run_name1 , run_name2, log_dir, ex_num, logits_used, display_combined=False)
        #f.close()



def print_logits_of_example(
    model_name='117M',
    run_names=('kdrama_finetune',),
    seed=None,
    ex_num='gen_court_kdrama_only_kdrama',
    batch_size=1,
    length=None,
    temperature=1,
    top_k=40,
    top_p=0.0,
    use_random=False,
    use_swap=False,
    raw_text=kdrama_sents
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
    losses = {}
    for run_name in run_names:
        losses[run_name] = []
    sents = sent_tokenize(raw_text)
    f = open('/home/twister/Dropbox (GaTech)/caras_graphs/{}_{}_{}.txt'.format(ex_num, len(sents), '{}'.format('_'.join(run_names))),'w', encoding='utf-8')
    f.writelines(sents)
    f.close()

    for run_name in run_names:
        for sent in sents:
            context_tokens = enc.encode(sent)
            #cont = [context_tokens for _ in range(batch_size)]
            with tf.Session(graph=tf.Graph()) as sess:
                context = tf.placeholder(tf.int32, [batch_size, None])
                np.random.seed(seed)
                tf.set_random_seed(seed)

                output = sample.return_logits(
                    hparams=hparams, run_name=run_name,
                    context=context,
                    batch_size=batch_size,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_random=use_random,
                    use_swap=use_swap
                )

                saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name in v.name])
                ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name))
                saver1.restore(sess, ckpt1)
                #next_outputs = sess.run(output)
                next_outputs = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })
                logits = next_outputs[0]
                nums = context_tokens #[int(context_tokens[0][z]) for z in range(context_tokens.shape[0])]
                probs = remove_min_max([logits[0][0][num] for num in nums])
                losses[run_name].append(sum(probs)/len(probs))
    create_graphs.create_sentence_chart_not_gen(losses, ex_num, run_names, len(sents))


if __name__ == '__main__':
    fire.Fire(print_logits_of_example)

