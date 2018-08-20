import tensorflow as tf
import numpy as np
import cifar10
# from google.colab import files


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def batch_getter(data, labels, batch_s):
    assert data.shape[0] == labels.shape[0]
    train_data_size = data.shape[0]
    data_per_batch = train_data_size // batch_s
    training_batches = np.split(data, data_per_batch)
    labels_batches = np.split(labels, data_per_batch)
    return np.array(training_batches), np.array(labels_batches)


class QinCapsNet:
    def __init__(self, caps1_n_maps, caps1_n_dims, caps2_n_caps, caps2_n_dims, n_epochs, batch_size, restore_checkpoint,
                 init_sigma=0.1):
        print("ATTENTION: ", " caps1_n_maps: ", caps1_n_maps, " caps1_n_dims: ", caps1_n_dims, " caps2_n_dims: ",
              caps2_n_dims)
        tf.reset_default_graph()

        self.caps1_n_maps = caps1_n_maps
        self.caps1_n_caps = caps1_n_maps * 8 * 8
        self.caps1_n_dims = caps1_n_dims

        self.caps2_n_caps = caps2_n_caps
        self.caps2_n_dims = caps2_n_dims

        self.init_sigma = init_sigma

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.restore_checkpoint = restore_checkpoint

        tf.reset_default_graph()

        np.random.seed(42)
        tf.set_random_seed(42)

        self.load_data()

    def load_data(self):
        cifar10.maybe_download_and_extract()
        self.images_train, self.cls_train, self.labels_train = cifar10.load_training_data()
        self.images_test, self.cls_test, self.labels_test = cifar10.load_test_data()

    def create_a_net(self):
        conv1_params = {
            "filters": 256,
            "kernel_size": 9,
            "strides": 1,
            "padding": "valid",
            "activation": tf.nn.relu,
        }

        conv2_params = {
            "filters": self.caps1_n_maps * self.caps1_n_dims,  # 256 convolutional filters
            "kernel_size": 9,
            "strides": 2,
            "padding": "valid",
            "activation": tf.nn.relu
        }

        X = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name="X")

        conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
        conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)

        caps1_raw = tf.reshape(conv2, [-1, self.caps1_n_caps, self.caps1_n_dims], name="caps1_raw")

        caps1_output = squash(caps1_raw, name="caps1_output")

        W_init = tf.random_normal(
            shape=(1, self.caps1_n_caps, self.caps2_n_caps, self.caps2_n_dims, self.caps1_n_dims),
            stddev=self.init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")

        batch_size = tf.shape(X)[0]
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

        caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                               name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                           name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.caps2_n_caps, 1, 1],
                                     name="caps1_output_tiled")

        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                    name="caps2_predicted")

        raw_weights = tf.zeros([batch_size, self.caps1_n_caps, self.caps2_n_caps, 1, 1],
                               dtype=np.float32, name="raw_weights")

        routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

        weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                           name="weighted_predictions")

        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                     name="weighted_sum")

        caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

        caps2_output_round_1_tiled = tf.tile(
            caps2_output_round_1, [1, self.caps1_n_caps, 1, 1, 1],
            name="caps2_output_round_1_tiled")

        agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                              transpose_a=True, name="agreement")

        raw_weights_round_2 = tf.add(raw_weights, agreement,
                                     name="raw_weights_round_2")

        routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                                dim=2,
                                                name="routing_weights_round_2")
        weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                                   caps2_predicted,
                                                   name="weighted_predictions_round_2")
        weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                             axis=1, keep_dims=True,
                                             name="weighted_sum_round_2")
        caps2_output_round_2 = squash(weighted_sum_round_2,
                                      axis=-2,
                                      name="caps2_output_round_2")

        caps2_output = caps2_output_round_2

        y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

        y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")

        y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

        y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

        m_plus = 0.9
        m_minus = 0.1
        lambda_ = 0.5

        T = tf.one_hot(y, depth=self.caps2_n_caps, name="T")

        caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                                      name="caps2_output_norm")

        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                      name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                                   name="present_error")

        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                     name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                                  name="absent_error")

        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
                   name="L")

        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

        mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

        reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                         lambda: y,  # if True
                                         lambda: y_pred,  # if False
                                         name="reconstruction_targets")

        reconstruction_mask = tf.one_hot(reconstruction_targets,
                                         depth=self.caps2_n_caps,
                                         name="reconstruction_mask")

        # reconstruction_mask_reshaped: (?, 1, 10, 1, 1)
        reconstruction_mask_reshaped = tf.reshape(
            reconstruction_mask, [-1, 1, self.caps2_n_caps, 1, 1],
            name="reconstruction_mask_reshaped")

        caps2_output_masked = tf.multiply(
            caps2_output, reconstruction_mask_reshaped,
            name="caps2_output_masked")

        decoder_input = tf.reshape(caps2_output_masked,
                                   [-1, self.caps2_n_caps * self.caps2_n_dims],
                                   name="decoder_input")


        n_hidden1 = 512
        n_hidden2 = 1024
        n_output = 32 * 32 * 3

        with tf.name_scope("decoder"):
            hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                      activation=tf.nn.relu,
                                      name="hidden1")
            hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                      activation=tf.nn.relu,
                                      name="hidden2")
            decoder_output = tf.layers.dense(hidden2, n_output,
                                             activation=tf.nn.sigmoid,
                                             name="decoder_output")
        X_flat = tf.reshape(X, [-1, n_output], name="X_flat")
        squared_difference = tf.square(X_flat - decoder_output,
                                       name="squared_difference")
        reconstruction_loss = tf.reduce_mean(squared_difference,
                                             name="reconstruction_loss")

        alpha = 0.0005
        loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")

        correct = tf.equal(y, y_pred, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss, name="training_op")

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        training_batched_data, training_batched_labels = batch_getter(self.images_train, self.cls_train, self.batch_size)
        testing_batched_data, testing_batched_labels = batch_getter(self.images_test, self.cls_test, self.batch_size)

        best_loss_val = np.infty

        save_suffix = '_' + str(self.caps1_n_maps) + '_' + str(self.caps1_n_dims) + '_' + str(self.caps2_n_dims)

        checkpoint_path = "./generated_files/qin_capsule_network.ckpt" + save_suffix

        with tf.Session() as sess:
            if self.restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
                print("Restored")
                saver.restore(sess, checkpoint_path)
            else:
                init.run()

            # for epoch in range(1):
            for epoch in range(self.n_epochs):
                print("epoch: ", epoch)

                for batch_i in range(len(training_batched_data)):
                # for batch_i in range(1):
                    if batch_i % 50 == 0:
                        print("batch: ", batch_i)
                    _, loss_train = sess.run(
                        [training_op, loss],
                        feed_dict={X: training_batched_data[batch_i].reshape([-1, 32, 32, 3]),
                                   y: training_batched_labels[batch_i],
                                   mask_with_labels: True})

                # At the end of each epoch,
                # measure the validation loss and accuracy:
                loss_vals = []
                acc_vals = []

                for batch_i in range(len(testing_batched_data)):
                # for batch_i in range(1):
                    loss_val, acc_val = sess.run(
                        [loss, accuracy],
                        feed_dict={X: testing_batched_data[batch_i].reshape([-1, 32, 32, 3]),
                                   y: testing_batched_labels[batch_i]})
                    loss_vals.append(loss_val)
                    acc_vals.append(acc_val)
                    if batch_i % 20 == 0:
                        print("Current loss: ", loss_val)
                        print("Current acc: ", acc_val)
                loss_val = np.mean(loss_vals)
                acc_val = np.mean(acc_vals)
                print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                    epoch + 1, acc_val * 100, loss_val,
                    " (improved)" if loss_val < best_loss_val else ""))

                # And save the model if it improved:
                if loss_val < best_loss_val:
                    save_path = saver.save(sess, checkpoint_path)
                    best_loss_val = loss_val
                    print("Model saved in path: %s" % save_path)

            tmp_file_name = './generated_files/' + str(self.caps1_n_maps) + '_' + str(self.caps1_n_dims) + '_' + str(self.caps2_n_dims) + \
                            '.txt'
            with open(tmp_file_name, 'a') as out:
                out.write(str(acc_val))
            return acc_val

            # files.download(checkpoint_path + '.data-00000-of-00001')
            # files.download(checkpoint_path + '.index')
            # files.download(checkpoint_path + '.meta')
