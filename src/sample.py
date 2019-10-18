import tensorflow as tf
import random

import model, encoder
import os, json


w1 = 0.5
w2 = 0.5
change = 0
inc = True


def pick_top_k_combined(logits, logits1, logits2, k):
    picks0 = top_k_logits(logits, k)
    picks1 = top_k_logits(logits1, k)
    picks2 = top_k_logits(logits2, k)
    randy = random.randint(0, 3)
    if randy == 0.0:
        return picks0
    elif randy == 1.0:
        return picks1
    else:
        return picks2
    #return picks0 + picks1 + picks2

def top_k_minus_vanilla(logits, vanilla, k):
    picks0 = top_k_logits(logits, k)
    picks1 = top_k_logits(vanilla, k)
    return picks0 - picks1*0.5

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )

def top_k_logits_softer(logits, k):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype)* 1e-10,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )

# def top_k_logits_combined(logits1, logits2, k):
#     logits =
#     if k == 0:
#         # no truncation
#         return logits
#
#     def _top_k():
#         values, _ = tf.nn.top_k(logits, k=k)
#         min_values = values[:, -1, tf.newaxis]
#         return tf.where(
#             logits < min_values,
#             tf.ones_like(logits, dtype=logits.dtype) * -1e10,
#             logits,
#         )
#     return tf.cond(
#        tf.equal(k, 0),
#        lambda: logits,
#        lambda: _top_k(),
#     )
#


def top_p_logits_combined(next_outputs, temperature, p):
    with tf.variable_scope('top_p_logits'):
        # logits_sort = tf.sort(logits, direction='DESCENDING')
        # probs_sort = tf.nn.softmax(logits_sort)
        # probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        # logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort) * 1000)  # [batchsize, vocab]
        # min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True)  # [batchsize, 1]

        #logits = tf.nn.softmax(next_outputs['logits1']) * w1 + tf.nn.softmax(next_outputs['logits2']) * w2
        logits1 = next_outputs['logits1'][:, -1, :] / tf.to_float(temperature)
        logits2 = next_outputs['logits2'][:, -1, :] / tf.to_float(temperature)
        logits_sort1 = tf.sort(logits1, direction='DESCENDING')
        logits_sort2 = tf.sort(logits2, direction='DESCENDING')

        probs_sort1 = tf.nn.softmax(logits_sort1)
        probs_sort2 = tf.nn.softmax(logits_sort2)

        probs_sums1 = tf.cumsum(probs_sort1, axis=1, exclusive=True)
        probs_sums2 = tf.cumsum(probs_sort2, axis=1, exclusive=True)

        logits_masked1 = tf.where(probs_sums1 < p, logits_sort1, tf.ones_like(logits_sort1)*1000) # [batchsize, vocab]
        logits_masked2 = tf.where(probs_sums2 < p, logits_sort2, tf.ones_like(logits_sort2)*1000) # [batchsize, vocab]

        min_logits1 = tf.reduce_min(logits_masked1, axis=1, keepdims=True) # [batchsize, 1]
        min_logits2 = tf.reduce_min(logits_masked2, axis=1, keepdims=True) # [batchsize, 1]

        res1 = tf.where(
            logits1 < min_logits1,
            tf.ones_like(logits1, dtype=logits1.dtype) * -1e10,
            logits1,
        )

        res2 = tf.where(
            logits2 < min_logits2,
            tf.ones_like(logits2, dtype=logits2.dtype) * -1e10,
            logits2,
        )
        return res1*w1 + res2*w2


def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if top_p > 0.0:
                logits = top_p_logits(logits, p=top_p)
            else:
                logits = top_k_logits(logits, k=top_k)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False
        )

        return tokens



def sample_sequence_combined(*, hparams, length, run_name1='', run_name2='', start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, top_k_combined=0, weight1=0.5, weight2=0.5, use_random=False, use_swap=False, use_f1=False, inc=False, use_fifty_one=True, debug=True, use_vanilla=False):
    # if weight1 >= 1.0:
    #     inc = False
    global w1
    global w2
    w1 = weight1
    w2 = weight2
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past1=None, past2=None, past_vanilla=None, we1=weight1, we2=weight2):
        lm_output = model.combined_model(hparams=hparams, scope1=run_name1, scope2=run_name2, X=tokens, past1=past1, past2=past2, reuse=tf.AUTO_REUSE, weight1=we1, weight2=we2)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents1 = lm_output['present1']
        presents1.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        presents2 = lm_output['present2']
        presents2.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        #if use_vanilla:
        lm_vanilla = model.model(hparams=hparams, X=tokens, past=past_vanilla, reuse=tf.AUTO_REUSE)
        presents_vanilla = lm_vanilla['present']
        presents_vanilla.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'logits1': lm_output['logits1'][:, :, :hparams.n_vocab],
            'logits2': lm_output['logits2'][:, :, :hparams.n_vocab],
            'logits_vanilla': lm_vanilla['logits'][:, :, :hparams.n_vocab],
            'presents1': presents1,
            'presents2': presents2,
            'presents_vanilla': presents_vanilla
        }
        # else:
        #     return {
        #         'logits': logits,
        #         'logits1': lm_output['logits1'][:, :, :hparams.n_vocab],
        #         'logits2': lm_output['logits2'][:, :, :hparams.n_vocab],
        #         'presents1': presents1,
        #         'presents2': presents2,
        #     }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past1, past2, past_vanilla, prev, output):
            global inc
            global change
            global w1
            global w2
            if use_random:
                w1 = weight_random()
                w2 = 1 - w1
            elif use_swap:
                w1 = weight_swap(w1)
                w2 = 1 - w1
            elif use_f1:
                if w1 == 1.0:
                    inc = False
                if w1 == 0.0:
                    inc = True
                weight_function1(w1, inc)
                # print("**" + str(w1))
            elif use_fifty_one:
                global change
                #if tf.math.greater(tf.size(output), tf.constant(int(length/2))):
                if change >= 1:
                    w1 = 1.0
                    w2 = 0.0
                else:
                    w1 = 0.5
                    w2 = 0.5
                    change += 1
            next_outputs = step(hparams, prev[:, tf.newaxis], past1=past1, past2=past2, past_vanilla=past_vanilla, we1=w1, we2=w2)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            logits1 = next_outputs['logits1'][:, -1, :] / tf.to_float(temperature)
            logits2 = next_outputs['logits2'][:, -1, :] / tf.to_float(temperature)
            logits_vanilla = next_outputs['logits_vanilla'][:, -1, :] / tf.to_float(temperature)
            # logits1 = tf.nn.softmax(next_outputs['logits1'])[:, -1, :] #/ tf.to_float(temperature)
            # logits2 = tf.nn.softmax(next_outputs['logits2'])[:, -1, :] #/ tf.to_float(temperature)

            if top_p > 0.0:
                logits = top_p_logits_combined(next_outputs, temperature, p=top_p)
            elif use_vanilla:
                #logits_vanilla = next_outputs['logits_vanilla'][:, -1, :] / tf.to_float(temperature)
                logits = top_k_minus_vanilla(logits, logits_vanilla, k=top_k)
            elif top_k_combined > 0.0:
                logits = pick_top_k_combined(logits, logits1, logits2, top_k)
            else:
                logits = top_k_logits(logits, k=top_k)
                if debug:
                    #logits1_idxs = top_k_logits_softer(logits1, k=top_k)
                    #logits2_idxs = top_k_logits_softer(logits2, k=top_k)
                    tf.summary.histogram(run_name1, logits1)
                    tf.summary.histogram(run_name2, logits2)

            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                    tf.concat([past1, next_outputs['presents1']], axis=-2),
                    tf.concat([past2, next_outputs['presents2']], axis=-2),
                    tf.concat([past2, next_outputs['presents_vanilla']], axis=-2),
                    tf.squeeze(samples, axis=[1]),
                    tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            return True

        _, _, _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length,
            loop_vars=[
                context_output['presents1'],
                context_output['presents2'],
                context_output['presents_vanilla'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens


def return_logits(*, hparams, run_name='', start_token=None, batch_size=None, context=None, temperature=1, top_k_combined=0, top_k=0, top_p=0.0, use_random=True, use_swap=False):


    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, scope=run_name, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            global w1
            global w2
            if use_random:
                w1 = weight_random()
                w2 = 1 - w1
            elif use_swap:
                w1 = weight_swap(w1)
                w2 = 1 - w1
                # print("**" + str(w1)
            
            next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            return [
                next_outputs['logits']
            ]

        def cond(*args):
            return True
        out = body(context_output['presents'], context[:, -1], context)

        return out


def return_combined_logits(*, hparams, length, run_name1='', run_name2='',
                           start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_k_combined=0, top_p=0.0,
                           weight1=0.5, weight2=0.5, use_random=False, use_swap=False, use_f1=False, inc=False,
                           use_fifty_one=True, debug=True, logits_used=0, display_logits=False):
    # if weight1 >= 1.0:
    #     inc = False
    global w1
    global w2
    w1 = weight1
    w2 = weight2
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    def step(hparams, tokens, past1=None, past2=None, we1=weight1, we2=weight2):
        lm_output = model.combined_model(hparams=hparams, scope1=run_name1, scope2=run_name2, X=tokens, past1=past1, past2=past2, reuse=tf.AUTO_REUSE, weight1=we1, weight2=we2)
        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents1 = lm_output['present1']
        presents1.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        presents2 = lm_output['present2']
        presents2.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        #if use_vanilla:
        return {
            'logits': logits,
            'logits1': lm_output['logits1'][:, :, :hparams.n_vocab],
            'logits2': lm_output['logits2'][:, :, :hparams.n_vocab],
            'presents1': presents1,
            'presents2': presents2,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past1, past2, prev, output):
            global inc
            global change
            global w1
            global w2
            if use_random:
                w1 = weight_random()
                w2 = 1 - w1
            elif use_swap:
                w1 = weight_swap(w1)
                w2 = 1 - w1
            elif use_f1:
                if w1 == 1.0:
                    inc = False
                if w1 == 0.0:
                    inc = True
                weight_function1(w1, inc)
                # print("**" + str(w1))
            elif use_fifty_one:
                global change
                #if tf.math.greater(tf.size(output), tf.constant(int(length/2))):
                if change >= 1:
                    w1 = 1.0
                    w2 = 0.0
                else:
                    w1 = 0.5
                    w2 = 0.5
                    change += 1
            next_outputs = step(hparams, prev[:, tf.newaxis], past1=past1, past2=past2, we1=w1, we2=w2)
            logits = next_outputs['logits'][:, -1, :] / tf.to_float(temperature)
            #logits1 = tf.nn.softmax(next_outputs['logits1'])[:, -1, :] / tf.to_float(temperature)
            #logits2 = tf.nn.softmax(next_outputs['logits2'])[:, -1, :] / tf.to_float(temperature)

            logits1 = next_outputs['logits1'][:, -1, :]  / tf.to_float(temperature)
            logits2 = next_outputs['logits2'][:, -1, :]  / tf.to_float(temperature)
            if top_k_combined > 0:
                lu = logits
            elif logits_used == 0:
                lu = logits
            elif logits_used == 1:
                lu = logits1
            elif logits_used == 2:
                lu = logits2
            else:
                lu = logits
            if False:#top_p > 0.0:
                logits = top_p_logits_combined(next_outputs, temperature, p=top_p)
                log = {}
            else:
                if display_logits:
                    log = {
                        'logits1': logits1,
                        'logits2': logits2,
                         'logits': logits,
                    }
                else:
                    log = {
                        'logits1': tf.nn.softmax(logits1),
                        'logits2': tf.nn.softmax(logits2),
                        'logits': tf.nn.softmax(logits),
                    }
                if top_k_combined > 0.0:
                    logits = pick_top_k_combined(logits, logits1, logits2, top_k)
                else:
                    logits = top_k_logits(lu, k=top_k)

                #if debug:
                    #logits1_idxs = top_k_logits(logits1, k=top_k)
                    #logits2_idxs = top_k_logits(logits2, k=top_k)
                    #tf.summary.histogram(run_name1, logits1)
                    #tf.summary.histogram(run_name2, logits2)


            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past1, next_outputs['presents1']], axis=-2),
                tf.concat([past2, next_outputs['presents2']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                log
            ]

        def cond(*args):
            return True

        i = 0
        p1 = context_output['presents1']
        p2 = context_output['presents2']
        previous = context[:, -1]
        o = context
        out_log = {}
        while i < length:
            p1, p2, previous, o, log = body(p1, p2, previous, o)
            out_log[i] = log
            i += 1
        # _, _, tokens = tf.while_loop(
        #     cond=cond, body=body,
        #     maximum_iterations=length,
        #     loop_vars=[
        #         context_output['presents'],
        #         context[:, -1],
        #         context,
        #     ],
        #     shape_invariants=[
        #         tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
        #         tf.TensorShape([batch_size]),
        #         tf.TensorShape([batch_size, None]),
        #     ],
        #     back_prop=False,
        # )
        return out_log, o


def weight_random():
    w1 = random.uniform(0, 1)
    return w1


def weight_swap(weight1):
    return 1 - weight1

def weight_one():
    return 1.0

def reset_to_default():
    global change
    global inc
    inc = True
    change = 0


def weight_function1(weight1, incr):
    if weight1 == 1.0:
        w1 = weight1 - 0.1
    elif incr:
        w1 = weight1 + 0.1 if weight1 + 0.1 < 1.0 else 1.0
    else:
        w1 = weight1 - 0.1 if weight1-0.1 > 0 else 0
    return w1


# if __name__ == '__main__':
#     model_name = '117M'
#     run_name1 = 'brown_romance'
#     run_name2 = 'cornell_supreme'
#     seed = None
#     nsamples = 15
#     batch_size = 1
#     length = None
#     temperature = 2
#     top_k = 40
#     top_p = 0.0
#     weight1 = 0.5
#     weight2 = 0.5
#     use_random = False
#     use_swap = False
#     enc = encoder.get_encoder(model_name)
#     hparams = model.default_hparams()
#     with open(os.path.join('models', model_name, 'hparams.json')) as f:
#         hparams.override_from_dict(json.load(f))
#
#     if length is None:
#         length = hparams.n_ctx
#     elif length > hparams.n_ctx:
#         raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
#     out_log, tokens = return_combined_logits(
#         hparams=hparams, run_name1=run_name1, run_name2=run_name2,
#         length=length,
#         start_token=enc.encoder['<|endoftext|>'],
#         batch_size=batch_size,
#         temperature=temperature,
#         top_k=top_k,
#         top_p=top_p,
#         weight1=weight1,
#         weight2=weight2,
#         use_random=use_random,
#         use_swap=use_swap
#     )
#     tokens = tokens[:, 1:]
#
#     print(out_log[0]['logits1'].shape)
#     print(out_log[0]['logits1_idxs'].shape)


