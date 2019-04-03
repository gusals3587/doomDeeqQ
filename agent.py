from hyperparam import *

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, identifier="DQNetwork"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(identifier):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            """
            First convent:
            CNN -> BatchNormalization -> ELU
            """

            self.conv1 = tf.layers.conv2d(
                    inputs = self.inputs_,
                    filters=32,
                    kernel_size = [8,8],
                    strides = [4, 4],
                    padding = "VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name = "conv1"
                    )

            self.conv1_batchnorm = tf.layers.batch_normalization(
                    self.conv1,
                    training = True,
                    epsilon = 1e-5,
                    name = 'batch_norm1'
                    )

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

            """
            Second convnet:
            CNN -> BatchNormalization -> ELU
            """

            self.conv2 = tf.layers.conv2d(
                    inputs = self.conv1_out,
                    filters = 64,
                    kernel_size = [4, 4],
                    strides = [2, 2],
                    padding = "VALID",
                    kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                    name = "conv2"
                    )

            self.conv2_batchnorm = tf.layers.batch_normalization(
                    self.conv2,
                    training = True,
                    epsilon = 1e-5,
                    name = 'batch_normal'
                    )

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")

            """
            Third convnet:
            CNN -> BatchNormalization -> ELU
            """

            self.conv3 = tf.layers.conv2d(
                    inputs = self.conv2_out,
                    filters = 128,
                    kernel_size = [4, 4],
                    strides = [2, 2],
                    padding = "VALID",
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    name="conv3"
                    )

            self.conv3_batchnorm = tf.layers.batch_normalization(
                    self.conv3,
                    training = True,
                    epsilon = 1e-5,
                    name = "batch_norm3"
                    )

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            # self.conv3_out -> [4 - 1, 4 - 1, 128]

            self.flatten = tf.layers.flatten(self.conv3_out)
            # self.flatten -> [1152] 

            # fc == fully connected
            self.fc = tf.layers.dense(
                    inputs = self.flatten,
                    units = 512,
                    activation = tf.nn.elu,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="fc1"
                    )

            self.output = tf.layers.dense(
                    inputs = self.fc,
                    units = self.action_size,
                    activation = None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    )
            

            self.actions_ = tf.placeholder(tf.float32, [None, 3], name="actions")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        
        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)

        return [self.buffer[i] for i in index]
