# https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96

import sys, getopt

import tensorflow as tf

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'


def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)


def main():
    run_name = 'strength_training2'
    checkpoint_dir = 'checkpoint/{}'.format(run_name)
    replace_from = 'model/'
    replace_to = '{}/'.format(run_name)
    add_prefix = None
    dry_run = False
    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main()
    #main(sys.argv[1:])

