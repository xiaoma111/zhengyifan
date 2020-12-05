from itertools import permutations
import tensorflow as tf
import random
import numpy as np
import pdb

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# tf.Session(config=config)

# from keras.backend.tensorflow_backend import set_session
# set_session(tf.Session(config=config))


class DeepNN():
    def __init__(self, args, stepSize):
        # game params
        self.N = args.N
        self.K = args.K
        self.M = args.M
        self.Q = args.Q
        self.stepSize = stepSize
        self.numfilters1 = args.numfilters1# 整数,输出空间的维数(即卷积中的滤波器数).
        self.numfilters2 = args.numfilters2
        self.batchSize = args.batchSize
        self.l2_const = args.l2_const

        self.numMoves = self.Q ** self.stepSize#没走一步有多少种移动方式，此程序中一步，决定五个格子

        # bulid DNN
        ##tf.Graph()表示实例化一个用于tensorflow计算和表示用的数据流图，
        # 不负责运行计算。在代码中添加的操作和数据都是画在纸上的画，而图就是呈现这些画的纸。
        self.graph = tf.Graph()
        self.build_DNN()

        # initialization
        ##运算图，及神经网络
        self.sess = tf.Session(graph = self.graph)
        self.sess.run(tf.variables_initializer(self.graph.get_collection('variables')))
        # self.sess.run(tf.global_variables_initializer())

        # save and load Params
        #用来获取一个名称是‘key’的集合中的所有元素，返回的是一个列表
        self.saver = tf.train.Saver(self.graph.get_collection('variables'))
        self.cost_his = []

        # policy entropy & cross entropy
        #而策略熵和交叉熵
        self.PolicyEntropy = 0
        self.CrossEntropy = 0
        self.cntEntropy = 0

    def build_DNN(self):
        # Neural Net
        with self.graph.as_default():
            # input placeholders
            #placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
            # 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
            self.batchInput = tf.placeholder(tf.float32, shape = [None, self.M, self.N, self.K])#输入数据类型，形状
            self.dropRate = tf.placeholder(tf.float32)#下降率
            self.isTraining = tf.placeholder(tf.bool, name="is_training")#是否训练还是测试

            x_img = tf.reshape(self.batchInput, [-1, self.N, self.K, self.M])#哪一维使用了-1，那这一维度就不定义大小，而是根据你的数据情况进行匹配

            # Conv0
            #“valid”
            #valid即只对图像中像素点“恰好”与卷积层对齐的部分进行卷积。上图中一维演示中输入宽度为13，卷积宽度为6，每次前进5格，当进行到（12、13）
            # 的时候因为每次步伐为5，5>2，所以（12、13）就不进行卷积了，舍弃了最右边的这两个数。

            # “same”
            # same则不同，尽可能对原始的输入左右两边进行padding从而使卷积核刚好全部覆盖所有输入，当进行padding后如果输入的宽度为奇数则会在右边
            # 再padding一下（如上图15+1=16，右边两个pad，左边一个pad）。
            conv0 = tf.layers.conv2d(x_img, self.numfilters1, kernel_size=[3,3], padding='same')#这层创建了一个卷积核，将输入进行卷积来输出一个 tensor
            conv0 = tf.layers.batch_normalization(conv0, axis=-1, training=self.isTraining)#标准化
            conv0 = tf.nn.relu(conv0) # batchSize * N * K * self.numfilters1#激活函数设置

            # Conv1
            conv1 = tf.layers.conv2d(conv0, self.numfilters1, kernel_size=[3,3], padding='same')
            conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.isTraining)
            conv1 = tf.nn.relu(conv1) # batchSize * N * K * self.numfilters1

            # Conv2
            conv2 = tf.layers.conv2d(conv1, self.numfilters1, kernel_size=[3,3], padding='same')
            conv2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.isTraining)
            conv2 = tf.nn.relu(conv2) # batchSize * N * K * self.numfilters1

            # Conv3
            conv3 = tf.layers.conv2d(conv2, self.numfilters1, kernel_size=[3,3], padding='same')
            conv3 = tf.layers.batch_normalization(conv3, axis=-1, training=self.isTraining)
            conv3 = tf.nn.relu(conv3) # batchSize * N * K * self.numfilters1

            # Output PiVec
            x4 = tf.layers.conv2d(conv3, 2, kernel_size=[1,1], padding='same')
            x4 = tf.layers.batch_normalization(x4, axis=-1, training=self.isTraining)
            x4 = tf.nn.relu(x4) # batchSize * (N-2) * (K-2) * self.numfilters1

            x4_flat = tf.reshape(x4, [-1, 2 * (self.N) * (self.K)])


            x5 = tf.layers.dense(x4_flat, self.numfilters2)
            x5 = tf.layers.batch_normalization(x5, axis=1, training=self.isTraining)
            x5 = tf.nn.relu(x5)
            x5_drop = tf.layers.dropout(x5, rate=self.dropRate) # batchSize x 1024
            #Softmax简单的说就是把一个N*1的向量归一化为（0，1）之间的值，由于其中采用指数运算，使得向量中数值较大的量特征更加明显。
            self.piVec = tf.nn.softmax(tf.layers.dense(x5_drop, self.numMoves))

            # Output zValue
            y4 = tf.layers.conv2d(conv3, 1, kernel_size=[1,1], padding='same')
            y4 = tf.layers.batch_normalization(y4, axis=-1, training=self.isTraining)
            y4 = tf.nn.relu(y4) # batchSize * (N-2) * (K-2) * self.numfilters1

            y4_flat = tf.reshape(y4, [-1, 1 * (self.N) * (self.K)])#平铺


            y5 = tf.layers.dense(y4_flat, int(self.numfilters2/2))#全连接层，得到一半的特征数
            y5 = tf.layers.batch_normalization(y5, axis=1, training=self.isTraining)
            y5 = tf.nn.relu(y5)
            y5_drop = tf.layers.dropout(y5, rate=self.dropRate) # batchSize x 1024

            self.zValue = tf.nn.tanh(tf.layers.dense(y5_drop, 1)) #模型得到的结果，得到一个特征数，对特征数使用tanh求值

            # calculate loss 计算模型结果与真实结果之间的误差
            self.targetPi = tf.placeholder(tf.float32, shape=[None, self.numMoves])
            self.targetZ = tf.placeholder(tf.float32, shape=[None, 1])
            self.lr = tf.placeholder(tf.float32)#下降率
            #用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值与和，主要用作降维或者计算tensor（图像）的平均值。
            self.loss_pi = tf.reduce_mean(-tf.reduce_sum(self.targetPi * tf.log(self.piVec), 1))
            # self.loss_pi =  tf.losses.softmax_cross_entropy(self.targetPi, self.piVec)
            self.loss_v = tf.losses.mean_squared_error(self.targetZ, self.zValue)

            # L2 regulization l2正则化，使得参数的权重衰减，防止过拟合
            self.allParams = self.graph.get_collection('variables')
            l2_params = 0
            for paramsList in self.allParams:
                l2_params += tf.nn.l2_loss(paramsList)
            l2_penalty = self.l2_const * (l2_params * 2)


            self.total_loss = self.loss_pi + self.loss_v + l2_penalty
            #关于tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练
            # 操作之前完成的操作，并配合tf.control_dependencies函数使用。
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss)

    #残差网络
    def residual_block(self, input_layer, output_channel):

        conv1 = tf.layers.batch_normalization(input_layer, axis=-1, training=self.isTraining)
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.layers.conv2d(conv1, output_channel, kernel_size=[3,3], padding='same')

        
        conv2 = tf.layers.batch_normalization(conv1, axis=-1, training=self.isTraining)
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.conv2d(conv2, output_channel, kernel_size=[3,3], padding='same')

        output = conv2 + input_layer
        return output

    def refresh_entropy(self):
        self.PolicyEntropy = 0
        self.CrossEntropy = 0
        self.cntEntropy = 0

    def output_entropy(self):
        return self.PolicyEntropy, self.CrossEntropy, self.cntEntropy

    #评估节点
    def evaluate_node(self, rawstate, selfplay):
        state = self.feature_extract(rawstate)
        piVec, zValue = self.sess.run([self.piVec, self.zValue],  
                                    feed_dict= {self.batchInput:state,
                                    self.dropRate: 0,
                                    self.isTraining: False,
                                    })
        return piVec, zValue

    #如何求解targetz
    def update_DNN(self, mini_batch, lr):
        # expand mini_batch
        state_batch = np.array([data[0] for data in mini_batch])#状态
        piVec_batch = np.array([data[1] for data in mini_batch])#pivec
        reward_batch = np.array([data[2] for data in mini_batch])[:,np.newaxis]#奖励
        state_batch = self.feature_extract(state_batch)
        _, batchLoss = self.sess.run([self.train_step, self.total_loss], 
                                feed_dict= {self.batchInput: state_batch,
                                self.dropRate: 0.3,
                                self.isTraining: True,
                                self.lr: lr,
                                self.targetPi: piVec_batch,
                                self.targetZ: reward_batch,
                                })
        self.cost_his.append(batchLoss)


    def feature_extract(self,state_batch):
        feature1 = np.copy(state_batch)
        feature1[feature1 == -1] = 0
        feature2 = np.copy(state_batch)
        feature2[feature2 == 1] = 0
        feature2[feature2 == -1] = 1
        feature3 = np.zeros(state_batch.shape)
        feature3[state_batch == 0] = 1

        state = np.reshape(np.hstack((feature1,feature2,feature3)),(len(feature1),self.M, self.N, self.K))
        return state

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    def saveParams(self, path):
        self.saver.save(self.sess, path)

    def loadParams(self, path):
        self.saver.restore(self.sess, path)

    def get_params(self):
        return self.graph.get_collection('variables'), self.sess.run(self.graph.get_collection('variables'))
