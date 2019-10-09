#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder, create_graphs

def interact_model(
    model_name='117M',
    run_name='brown_romance',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=20,
    temperature=1,
    top_k=0,
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
    run_name1='brown_romance',
    run_name2='scifi',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=25,
    temperature=1,
    top_k=40,
    top_p=0.0,
    weight1=0.0,
    weight2=1.0,
    use_random=False,
    use_swap=False,
    use_fifty_one=False,
    debug=True,
    logits_used=0,
    ex_num='ex_sent_start',
    display_logits=True,
    display_combined=False,
    repeat=5
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
    log_dir = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/logs'
    losses0 = []
    losses1 = []
    losses2 = []
    all_text = []
    for cnt in range(repeat):
        raw_text = ' '.join(all_text).replace('\n', '').replace('<|endoftext|>', '')
        with tf.Session(graph=tf.Graph()) as sess:
            np.random.seed(seed)
            tf.set_random_seed(seed)
            context = tf.placeholder(tf.int32, [batch_size, None])

            output = sample.return_combined_logits(
                hparams=hparams, run_name1=run_name1, run_name2=run_name2,
                length=length + 20*cnt,
                start_token=enc.encoder['<|endoftext|>'],
                batch_size=batch_size,
                temperature=temperature,
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
            #names = [run_name1, run_name2, 'combined']
            while nsamples == 0 or generated < nsamples:
                # feed in sentence from before
                context_tokens = enc.encode(raw_text)
                #print('\nnew: {} len_context: {}\n'.format(raw_text, len(context_tokens)))
                out_log, out = sess.run(output,feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })
                out = out[:, len(context_tokens):]

                # # calculate the sentence losses
                # output0 = model.combined_model(hparams=hparams, X=context, scope1=run_name1, scope2=run_name2)
                # t1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                #         labels=context[:, 1:], logits=output0['logits'][:, :-1])
                # print('******t1 shape {}'.format(t1.shape))
                # loss0 = tf.reduce_mean(t1)
                # loss0_summary = tf.summary.scalar('loss0', loss0)
                # print('******loss0 shape {}'.format(loss0.shape))
                # output1 = model.model(hparams=hparams, X=context, scope=run_name1)
                # loss1 = tf.reduce_mean(
                #     tf.nn.sparse_softmax_cross_entropy_with_logits(
                #         labels=context[:, 1:], logits=output1['logits'][:, :-1]))
                # loss1_summary = tf.summary.scalar('loss1', loss1)
                # output2 = model.model(hparams=hparams, X=context, scope=run_name2)
                # loss2 = tf.reduce_mean(
                #     tf.nn.sparse_softmax_cross_entropy_with_logits(
                #         labels=context[:, 1:], logits=output2['logits'][:, :-1]))
                # loss2_summary = tf.summary.scalar('loss2', loss2)


                # do the logits graph stuff
                # for i in out_log.keys():
                #     for j, logy in enumerate(['logits1', 'logits2', 'logits']):
                #         out_file1 = '{}/{}/{}/{}_{}/{}/{}_{}.json'.format(log_dir, ex_num, logits_used, run_name1, run_name2, names[j], i, logy)
                #         odict = {}
                #         of1 = open(out_file1, 'w', encoding='utf-8')
                #         logits = out_log[i][logy]
                #         for k in range(logits.shape[1]):
                #             val = logits[0][k]
                #             sym = enc.decoder[k]
                #             #of1.write('{},{}\n'.format(sym, val))
                #             odict[str(sym)] = float(val)
                #         json.dump(odict, of1)
                #         of1.close()
                # # out_file = 'logs/{}_{}/logits1/{}'
                for i in range(batch_size):
                    generated += batch_size
                    text = enc.decode(out[i])
                    all_text.append(text)
                    nums = [int(out[i][z]) for z in range(out[i].shape[0])]
                    loss0 = sum([out_log[max(out_log.keys())]['logits'][0][num] for num in nums])
                    loss1 = sum([out_log[max(out_log.keys())]['logits1'][0][num] for num in nums])
                    loss2 = sum([out_log[max(out_log.keys())]['logits2'][0][num] for num in nums])
                    losses0.append(loss0)
                    losses1.append(loss1)
                    losses2.append(loss2)
                    sample_str = '\n' + "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + '\n'
                    # f.write(sample_str)
                    # f.write(text)
                    print(sample_str)
                    print(text)
                #raw_text = ''

        # print("*****{}****".format(type(losses1[0])))
        # print("*****{}****".format(losses1[0].shape))
    losses_dict = {run_name1:losses1, run_name2:losses2, 'combined':losses0}
    text_file = '/home/twister/Dropbox (GaTech)/caras_graphs/{}_{}_{}_{}_{}_{}_{}.txt'.format(ex_num, repeat, logits_used, run_name1, run_name2, weight1, weight2)
    tfile = open(text_file, 'w', encoding='utf-8')
    all_text = [txt.replace('\n', '').replace('<|endoftext|>', '') for txt in all_text]
    tfile.write('\n'.join(all_text))
    # nums = [str(out[i][z]) for z in range(out[i].shape[0])]
    # tfile.write(' '.join(nums))
    tfile.close()
    #create_graphs.create_word_chart(model_name, run_name1 , run_name2, log_dir, ex_num, logits_used, display_combined=True)
    print('{} sents'.format(len(losses0)))
    create_graphs.create_sentence_chart(losses_dict, ex_num, run_name1, run_name2, logits_used, repeat, weight1, weight2, display_combined=display_combined)
    #f.close()

def interact_combined_model(
        model_name='117M',
        run_name1='brown_romance',
        run_name2='scifi',
        seed=None,
        nsamples=3,
        batch_size=1,
        length=300,
        temperature=1.2,
        top_k=50,
        top_p=0.0,
        weight1=0.5,
        weight2=0.5,
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

if __name__ == '__main__':
    fire.Fire(print_combined_sentences)
