#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
import json

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
    run_name1='brown_romance',
    run_name2='cornell_supreme',
    seed=None,
    nsamples=200,
    batch_size=1,
    length=30,
    temperature=1,
    top_k=40,
    top_p=0.0,
    weight1=0.5,
    weight2=0.5,
    use_random=False,
    use_swap=False,
    use_fifty_one=True,
    debug=True
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
    tester_body = open('testy_body.txt', 'a', encoding='utf-8')
    tester_body.write('calling sample_sequence\n')
    tester_body.close()
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
        if use_fifty_one:
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
        tester = open('testy_gen.txt', 'w', encoding='utf-8')

        testLines = []
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
    run_name1='brown_romance',
    run_name2='scifi',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=25,
    temperature=1,
    top_k=0,
    top_p=0.0,
    weight1=0.0,
    weight2=1.0,
    use_random=False,
    use_swap=False,
    use_fifty_one=False,
    debug=True
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

        output = sample.return_combined_logits(
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
            use_swap=use_swap,
        )

        saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name1 in v.name])
        ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name1))
        saver1.restore(sess, ckpt1)
        saver2 = tf.train.Saver([v for v in tf.all_variables() if run_name2 in v.name])
        ckpt2 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name2))
        saver2.restore(sess, ckpt2)
        generated = 0
        names = [run_name1, run_name2]
        ex_num = 'ex1'
        while nsamples == 0 or generated < nsamples:
            out_log, out = sess.run(output)
            for i in out_log.keys():
                for j, logy in enumerate(['logits1', 'logits2']):
                    out_file1 = 'logs/{}/{}_{}/{}/{}_{}.json'.format(ex_num, run_name1, run_name2, names[j], i, logy)
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
                text_file = 'logs/{}/{}_{}/text.txt'.format(ex_num, run_name1, run_name2)
                tfile = open(text_file, 'w', encoding='utf-8')
                tfile.write(text + '\n')
                nums = [str(out[i][z]) for z in range(out[i].shape[0])]
                tfile.write(' '.join(nums))
                tfile.close()

        #f.close()

if __name__ == '__main__':
    fire.Fire(print_combined_logits)
