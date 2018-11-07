import os
import tensorflow as tf
import numpy as np
dim_image = 4096
dim_hidden = 512

n_video_lstm_step = 40
n_capion_lstm_step = 20
n_frame_step =40

n_epoch = 1200
batch_size = 25
learning = 0.001

save_epoch = 1

# =======================================
# 相关文件路径
#========================================
video_train_data_path = '../data/ch_video_train_seg.csv'
video_test_data_path = '../data/ch_video_test_seg.csv'
video_feature_path = ''

save_path = '../save_1101'      # 模型保存路径
if not os.path.exists(save_path):
    os.makedirs(save_path+'/models')

loss_out_path = save_path+'loss.txt'        # loss值记录在此txt文件当中
loss_save_path = save_path+'save_loss.npy'  #？？？
lossing_img_path = save_path+'loss.image'       # loss值画图

wordtoix_path = save_path+'wordtoix.npy'
ixtoword_path = save_path+'ixtoword.npy'
bias_init_vector_path = save_path+'bias_init_vector.npy'
epoch_save_path = save_path+'now_epoch.npy'
word_vec_path = save_path+'./data/word_vec_list.npy'

my_log_path = save_path+'czy_log.txt'
my_np_txt = save_path+'czy_np.txt'
lag_out = open(my_log_path, 'w', encoding='utf-8')

out = open(save_path+'out_now.txt', 'w', encoding='utf8')


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_capton_lstm_step,
                 bias_init_vector=None, word_vec_list=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_capion_lstm_step
        with tf.device("/cpu:0"):               # 嵌入层使用cpu进行计算，避免占用gpu，但是好像说这样做会减慢速度
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden/2, state_is_tuple = False)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple = False)
        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_w')
        self.encode_image_b = tf.Variable(tf.random_uniform([dim_hidden]), name='encode_image_b')

        #
        with tf.device("/cpu:0"):
            if word_vec_list is not None:
                self.embed_word_W = tf.Variable(word_vec_list.transpose(), name='embed_word_W')
            else:
                self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='b')
        self.attention_W = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='attention_w')
        self.attention_b = tf.Variable(tf.random_uniform([batch_size, dim_hidden], -0.1, 0.1), name='attention_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [batch_size, n_video_lstm_step, dim_hidden])
        video_mark = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mark = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        state3 = tf.zeros([self.batch_size, self.lstm1.state_size])

        padding_lstm1 = tf.zeros([self.batch_size, self.dim_hidden])
        padding_lstm2 = tf.zeros([self.batch_size, self.dim_hidden*2])

        prob = []
        loss = 0.0

        # ============================================
        # endocing stage
        #=============================================

        with tf.variable_scope(tf.get_variable_scope) as scope:
            list_out1 = []
            list_out3 = []
            list_state1 = []
            list_state3 = []

            for i in range(n_video_lstm_step):
                with tf.variable_scope('lstm1'):

                    out1, state1 =self.lstm1(image_emb[:, i, :], state1)
