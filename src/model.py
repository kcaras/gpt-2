import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results


def combined_model(hparams, X, past1=None, past2=None, scope1='brown_romance', scope2='cornell_supreme', reuse=tf.AUTO_REUSE, weight1=0.6, weight2=0.4):
    results = {}
    with tf.variable_scope(scope1, reuse=reuse):
        batch1, sequence1 = shape_list(X)

        wpe1 = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte1 = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length1 = 0 if past1 is None else tf.shape(past1)[-2]
        h1 = tf.gather(wte1, X) + tf.gather(wpe1, positions_for(X, past_length1))

        # Transformer
        presents1 = []
        pasts1 = tf.unstack(past1, axis=1) if past1 is not None else [None] * hparams.n_layer
        assert len(pasts1) == hparams.n_layer
        for layer, past in enumerate(pasts1):
            h1, present1 = block(h1, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints1', h1)
            presents1.append(present1)
        results['present1'] = tf.stack(presents1, axis=1)
        #results['present1'] = tf.math.log(tf.math.exp(results['present1'])*weight1)
        h1 = norm(h1, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat1 = tf.reshape(h1, [batch1*sequence1, hparams.n_embd])
        logits1 = tf.matmul(h_flat1, wte1, transpose_b=True)
        logits1 = tf.reshape(logits1, [batch1, sequence1, hparams.n_vocab])
        results['logits1'] = logits1
        # print('********************************************\n')
        # tf.print(results['logits1'], [results['logits1']])
        # print('<!>!<!>!<>!<!>!<!>AFTER<!>!<!>!<>!<!>!<!>\n')
        #results['logits1'] = tf.math.log(tf.math.exp(logits1)*weight1)

    with tf.variable_scope(scope2, reuse=reuse):
        batch2, sequence2 = shape_list(X)

        wpe2 = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.01))
        wte2 = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                              initializer=tf.random_normal_initializer(stddev=0.02))
        past_length2 = 0 if past2 is None else tf.shape(past2)[-2]
        h2 = tf.gather(wte2, X) + tf.gather(wpe2, positions_for(X, past_length2))

        # Transformer
        presents2 = []
        pasts2 = tf.unstack(past2, axis=1) if past2 is not None else [None] * hparams.n_layer
        assert len(pasts2) == hparams.n_layer
        for layer, past in enumerate(pasts2):
            h2, present2 = block(h2, 'h%d' % layer, past=past, hparams=hparams)
            if layer == 10:
                tf.add_to_collection('checkpoints2', h2)
            presents2.append(present2)
        results['present2'] = tf.stack(presents2, axis=1)
        #results['present2'] = tf.math.log(tf.math.exp(results['present2'])*weight2)
        h2 = norm(h2, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat2 = tf.reshape(h2, [batch2 * sequence2, hparams.n_embd])
        logits2 = tf.matmul(h_flat2, wte2, transpose_b=True)
        logits2 = tf.reshape(logits2, [batch2, sequence2, hparams.n_vocab])
        results['logits2'] = logits2
        #results['logits2'] = tf.math.log(tf.math.exp(logits2)*weight2)
    results['logits'] = tf.nn.softmax(results['logits1'])*weight1 + tf.nn.softmax(results['logits2'])*weight2
    #results['present'] = tf.nn.softmax(results['present1'])*weight1 + tf.nn.softmax(results['present2'])*weight2
    #results['logits'] = logits1 #+ logits2
    #results['present'] = results['present1'] #+ results['present2']
    #results['logits'] = tf.math.log_prob(tf.math.exp(results['logits1'])*weight1 + tf.math.exp(results['logits2'])*weight2 + tf.math.exp(0.00000001))
    #results['present'] = tf.math.log(tf.math.exp(results['present1'])*weight1 + tf.math.exp(results['present2'])*weight2 + tf.math.exp(0.00000001))
    return results
