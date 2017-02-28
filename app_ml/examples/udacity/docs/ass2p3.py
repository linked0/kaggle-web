from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np

# reference
# https://discussions.udacity.com/t/assignment-3-problem-4-very-poor-accuracy-with-a-deeper-network/157742/7
# http://yaroslavvb.blogspot.kr/2011/09/notmnist-dataset.html?showComment=1391023266211#c8758720086795711595

# result
# test accuracy: 91.5%

pickle_file = './data/large-data/notMNIST2.pickle'

try:
    f = open(pickle_file, 'rb')
    data = pickle.load(f)
    f.close()
except Exception as e:
    print('unable to load file ', pickle_file, ':', e)

train_dataset = data['train_dataset']
train_labels = data['train_labels']
valid_dataset = data['valid_dataset']
valid_labels = data['valid_labels']
test_dataset = data['test_dataset']
test_labels = data['test_labels']

print('cleansed training data: ', train_dataset.shape, train_labels.shape)
print('cleansed validation data: ', valid_dataset.shape, valid_labels.shape)
print('cleansed test data: ', test_dataset.shape, test_labels.shape)

batch_size = 128
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set: %s, %s' % (train_dataset.shape, train_labels.shape))
print('Validation set: %s, %s' % (valid_dataset.shape, valid_labels.shape))
print('Test set: %s, %s' %(test_dataset.shape, test_labels.shape))

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


hidden_size = 400
num_steps = 10001
beta = 0.01
keep_prob = 0.5
initial_learning_rate = 0.5

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    weights_l1 = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_size]))
    biases_l1 = tf.Variable(tf.zeros([hidden_size]))
    logits_l1 = tf.matmul(tf_train_dataset, weights_l1) + biases_l1
    logits_l1 = tf.nn.relu(logits_l1)
    #     logits_l1 = tf.nn.dropout(logits_l1, keep_prob)

    weights_l2 = tf.Variable(tf.truncated_normal([hidden_size, num_labels]))
    biases_l2 = tf.Variable(tf.zeros([num_labels]))
    logits_l2 = tf.matmul(logits_l1, weights_l2) + biases_l2

    regularizers = tf.nn.l2_loss(weights_l1) + tf.nn.l2_loss(weights_l2)
    regularizers = 0.5 * regularizers * beta
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits_l2, tf_train_labels)) + regularizers

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                               1000, 0.98, staircase=True)
    #     optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_prediction = tf.nn.softmax(logits_l2)
    valid_layer1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_l1) + biases_l1)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_layer1, weights_l2) + biases_l2)
    test_layer1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_l1) + biases_l1)
    test_prediction = tf.nn.softmax(tf.matmul(test_layer1, weights_l2) + biases_l2)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 200 == 0):
            print('Minibatch loss at step: %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
