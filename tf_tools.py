import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def tf_checkpoint_to_model_dict(ckpt_name):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_name))
        saver.restore(sess, ckpt_name)
        t_vars = tf.trainable_variables()
        sess.as_default()
        model_vars = {}
        for var in t_vars:
            try:
                model_vars[var.name] = var.eval()
            except:
                print("For var={}, an exception occurred".format(var.name))
    return model_vars

def init_keras_model_from_ckpt(ckpt_name, keras_model, model_kargs):
    model_vars = tf_checkpoint_to_model_dict(ckpt_name)
    model = keras_model(**model_kargs)
    for lay in model.layers:
        if len(lay.get_weights())>0:
            print("Copy weights in", lay.name)
            lay.set_weights([model_vars['{}/kernel:0'.format(lay.name)], model_vars['{}/bias:0'.format(lay.name)]])

    return model