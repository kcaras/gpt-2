import os
import tensorflow as tf


def l1(run_name, orig_name):
    checkpoint_dir_orig = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/gpt-2/models/{}'.format(orig_name)
    checkpoint_dir_run = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/gpt-2/checkpoint/{}'.format(run_name)
    #checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    out_f = open('l1_values/{}_t2.txt'.format(run_name), 'w', encoding='utf-8')
    out_f.write('Running L1 on {}\n'.format(run_name))
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir_run):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir_run, var_name)
            var_orig = tf.contrib.framework.load_variable(checkpoint_dir_orig, var_name.replace(run_name, 'model'))

            normed = tf.norm(var - var_orig, ord=1)
            out_f.write('{}   {}\n'.format(var_name, sess.run(normed)))
            # Set the new name
            # new_name = var_name
            # if None not in [replace_from, replace_to]:
            #     new_name = new_name.replace(replace_from, replace_to)
            # if add_prefix:
            #     new_name = add_prefix + new_name
            #
            # if dry_run:
            #     print('%s would be renamed to %s.' % (var_name, new_name))
            # else:
            #     print('Renaming %s to %s.' % (var_name, new_name))
            #     # Rename the variable
            #     var = tf.Variable(var, name=new_name)
    out_f.close()
        # if not dry_run:
        #     # Save the variables
        #     saver = tf.train.Saver()
        #     sess.run(tf.global_variables_initializer())
        #     saver.save(sess, checkpoint.model_checkpoint_path)


def get_finetuned_L1(run_name, reuse=tf.AUTO_REUSE):
    f = open('/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/gpt-2/trained_vars_scopes.txt',
             'r', encoding='utf-8')
    scopes_sizes = f.readlines()
    f.close()
    out_f = open('l1_values/{}.txt'.format(run_name), 'w', encoding='utf-8')
    out_f.write('Running L1 on {}\n'.format(run_name))
    with tf.Session(graph=tf.Graph()) as sess:
        for ss in scopes_sizes:
            split = ss.replace('\n', '').split()
            scope = split[0]
            size = [val for val in split[1].replace('(', '').replace(')', '').split(',') if val]
            scope_name = scope.format(run_name).replace('\n', '')
            vname = scope.split('/')[-1]
            if vname == 'w':
                var = tf.get_variable(scope_name, size, initializer=tf.random_normal_initializer(stddev=0.02))

        ckpt1 = tf.train.latest_checkpoint(os.path.join('checkpoint', run_name))
        saver1 = tf.train.Saver([v for v in tf.all_variables() if run_name in v.name])
        saver1.restore(sess, ckpt1)
        with tf.variable_scope(run_name, reuse=reuse):
            for ss in scopes_sizes:
                split = ss.replace('\n', '').replace('{}/', '').split()
                scope = split[0]
                size = [val for val in split[1].replace('(', '').replace(')', '').split(',') if val]
                scope_name = scope.format(run_name).replace('\n', '')
                vname = scope.split('/')[-1]
                if vname == 'w':
                    var = tf.get_variable(scope_name, size)
                # elif vname == 'b':
                #     b = tf.get_variable(scope_name, [nf], initializer=tf.constant_initializer(0))
                # elif vname == 'g':
                    normed = tf.norm(var, ord=1)
                    out_f.write('{}   {}\n'.format(scope, sess.run(normed)))


def compare_l1(run_name1, original):
    dir = '/media/twister/04dc1255-e775-4227-9673-cea8d37872c7/humor_gen/caras_humor/gpt-2/l1_values/{}_t2.txt'
    f1 = open(dir.format(run_name1), 'r', encoding='utf-8')
    lines1 = f1.readlines()
    f1.close()
    # forig = open(dir.format(original), 'r', encoding='utf-8')
    # olines = forig.readlines()
    # forig.close()
    # od = {}
    fd = {}
    for line in lines1[1:]:
        split = line.strip().split()
        var_name = split[0]
        l1 = float(split[1])
        fd[var_name.replace('{}/'.format(run_name1), '')] = l1

    # for line in olines:
    #     split = line.strip().split()
    #     var_name = split[0]
    #     l1 = float(split[1])
    #     od[var_name.replace('model/', '')] = l1

    dsum = 0
    for vname in fd:
        val = fd[vname]
        #val = abs(od[vname] - fd[vname])
        #print('{}  diff --> {}'.format(vname, val))
        dsum += val
    print('{} total diff: {}'.format(run_name1, dsum))

if __name__ == '__main__':
    #l1('cornell_supreme', '117M')
    names = ['cornell_supreme', 'brown_romance', 'cornell_movies']
    for n in names:
        compare_l1(n, '117M')
