import pandas as pd
import numpy as np
import os
import tensorflow as tf
import time
from keras.preprocessing import sequence
# from gensim.models import Word2Vec

import matplotlib
matplotlib.use('Agg')  # 确定使用的后端
import matplotlib.pyplot as plt



#=======================================================================================
# Train Parameters
#=======================================================================================
dim_image = 4096  # 图片维度,即为输入的维度
dim_hidden = 512  # 隐藏层维度

n_video_lstm_step = 80  # 视频特征提起lstm维度
n_caption_lstm_step = 20  # 描述生成 lstm维度
n_frame_step = 80  # 视频帧数

n_epochs = 100  # 轮数
batch_size = 12  # batch大小
learning_rate = 0.001  # 学习率

save_epoch = 1  # 多少轮保存一个模型

#=======================================================================================
# relative file
#=======================================================================================

video_train_data_path = '../data/ch_video_train_seg.csv'
# video_train_data_path = '..\\data\\ch_add_train_seg.csv'
#video_train_data_path = '../data/ch_vec_train_seg.csv'

video_test_data_path = '../data/ch_video_test_seg.csv'
# video_test_data_path = '..\\data\\ch_add_test_seg.csv'
# video_test_data_path = '../data/ch_vec_test_seg.csv'

video_feature_path = 'E:/1我的东西/我的很多资料/研究生/python project/VideoCaptioning_att-master_czy/temp_RGB_feats'
# video_feature_path = 'D:E:/1我的东西/我的很多资料/研究生/python project/VideoCaptioning_att-master_czy/temp_RGB_feats'


# 保存该模型下的相关数据
save_path = '../save_1101/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_models_path = save_path + 'models/'

loss_out_path = save_path + 'loss.txt'  # loss输出保存
loss_save_path = save_path + 'save_loss.npy'  # loss值记录
loss_img_path = save_path + 'loss_img.png'

wordtoix_path = save_path + "wordtoix.npy"
ixtoword_path = save_path + 'ixtoword.npy'
bias_init_vector_path = save_path + "bias_init_vector.npy"
epoch_save_path = save_path + 'now_epoch.npy'
word_vec_path = save_path + './data/word_vec_list.npy'

my_log_path = save_path + 'czy_log.txt'
my_np_text = save_path + 'czy_np.txt'
log_out = open(my_log_path, 'w', encoding='utf-8')

out = open(save_path + 'out_now.txt', 'w', encoding='utf-8')


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step,
                 bias_init_vector=None, word_vec_list=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step
        with tf.device("/cpu:0"):           # 指定使用cpu进行计算
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')       # 这个变量的作用是什么
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden/2, state_is_tuple=False)         # 两端的编码器
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

        # 用cpu存储词嵌入，避免占显存
        with tf.device('/cpu:0'):
            if word_vec_list is not None:
                # 需要把词汇表转置
                self.embed_word_W = tf.Variable(word_vec_list.transpose(), name='embed_word_W')
            else:
                self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.attention_W = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1, 0.1), name='attention_W')

        self.decoder_W = tf.Variable(tf.random_uniform([batch_size, dim_hidden], -0.1, 0.1), name='decoder_W')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])        # 三维图像向量
        video_mark = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])  # caption为生成的文本，必须int，因为它是被用到词表里面取词的。
        caption_mark = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])  # [batch_size*n_video_lstm_step, dim_image]
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)  # [batch_size*n_video_lstm_step, dim_hidden]
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])  # [batch_size, n_video_lstm_step, dim_hidden]

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])  # [batch_size, dim_hidden*2]
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])  # [batch_size, dim_hidden*2]
        state3 = tf.zeros([self.batch_size, self.lstm1.state_size])  # [batch_size, dim_hidden*2]

        padding_lstm1 = tf.zeros([self.batch_size, self.dim_hidden])  # [batch_size, dim_hidden]
        padding_lstm2 = tf.zeros([self.batch_size, 2*self.dim_hidden])  # [batch_size, dim_hidden]

        probs = []
        loss = 0.0

        #=====================================================
        #  Encoding Stage
        #=====================================================
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            list_out1 = []
            list_out3 = []
            list_state1 = []
            list_state3 = []
            # 分别输入每个视频帧
            for i in range(0, n_video_lstm_step):
                # 后面的单元复用神经网络变量
                if i > 0:
                    tf.get_variable_scope().reuse_variables()       # 支持LSTM复用
                with tf.variable_scope('LSTM1'):

                    # 输入当前图片帧，获取输出1,状态1
                    out1, state1 = self.lstm1(image_emb[:, i, :], state1)  # 状态不断被下一个lstm使用  # out1--[batch_size, dim_hidden], state1--[batch_size, dim_hidden*2]
                    out3, state3 = self.lstm1(image_emb[:, n_video_lstm_step-i-1, :], state3)
                    list_out1.append(out1)
                    list_out3.append(out3)
                    list_state1.append(state1)
                    list_state3.append(state3)

                #左边为h，右边为state
            for i in range(0, n_video_lstm_step):
                out_1 = tf.concat((list_out1[i], list_out3[n_video_lstm_step-i-1]), axis=1)
                # #state_1 = tf.concat((state1, state3), axis=1)
                # state_1 = tf.concat((list_state1[n_video_lstm_step-1], list_state3[0]), axis=1)

                # 将所有的隐状态串联， 不断收集隐状态
                if i == 0:
                    hidden1 = tf.expand_dims(out_1, axis=1)  # [batch_size, 1, dim_hidden]
                else:
                    hidden1 = tf.concat([hidden1, tf.expand_dims(out_1, axis=1)], axis=1)  # [batch_size, 2->n_video_lstm_step, dim_hidden]

                with tf.variable_scope('LSTM2', reuse=tf.AUTO_REUSE):
                    # lstm2输入的宽度是lstm1的2倍
                    concat_temp = tf.concat([padding_lstm2, out_1], axis=1)  # concat_temp--[batch_size, dim_hidden*2+dim_hidden]
                    out2, state2 = self.lstm2(concat_temp, state2)  # 状态不断被下一个lstm使用  # out2--[batch_size, dim_hidden], state2--[batch_size, dim_hidden*2]


        #=====================================================
        #  Decoding Stage
        #=====================================================
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            # 分别输出每个描述
            for i in range(0, n_caption_lstm_step):
                # cpu下获取词汇表
                with tf.device('/cpu:0'):
                    now_caption = caption[:, i]  # [40]，占位符
                    # current_embed 固定谓前一位的标签，因为后面的标签后移了
                    current_embed = tf.nn.embedding_lookup(self.Wemb, now_caption) # 获取对应的词向量  # [batch_size, dim_hidden]
                tf.get_variable_scope().reuse_variables()

                #  此时无图像，用空白填充S
                with tf.variable_scope('LSTM1'):
                    out1, state1 = self.lstm1(padding_lstm1, state1)  # state1来自编码阶段的正向编码  out1--[batch_size, dim_hidden/2], state1--[batch_size, dim_hidden*2]
                    out3, state3 = self.lstm1(padding_lstm1, state3)  # state1来自编码阶段的反向编码  out1--[batch_size, dim_hidden/2], state1--[batch_size, dim_hidden*2]
                    out_1 = tf.concat((out1, out3), axis=1)
                # 计算文本向量
                # 对编码阶段的隐向量1进行reshape （多个out拼接）
                hidden1 = tf.reshape(hidden1, [-1, self.dim_hidden])  # [batch_size*n_video_step,dim_hidden]
                # 编码阶段的隐向量2 （最后一个out）
                hidden2 = tf.reshape(out2, [-1, 1])  # [batch_size*dim_hidden, 1]   #？？

                # 注意力机制的权重
                Wbilinear = tf.tile(self.attention_W, tf.stack([1, self.batch_size]))   # [dim_hidden, dim_hidden*batch_size]

                # 隐层1乘上batch_size的attention
                alpha = tf.matmul(hidden1, Wbilinear)  # [batch_size*n_video_step, dim_hidden*batch_size]

                # alpha 乘隐层2
                alpha = tf.matmul(alpha, hidden2)  # [batch_size*n_video_step, 1]

                # 以batch_size 隔开 alpha
                alpha = tf.reshape(alpha, [self.batch_size, -1])  # [batch_size, n_video_step]

                # 求sofrmax
                alpha = tf.nn.softmax(alpha)

                # 重新把alpha换回来
                alpha = tf.reshape(alpha, [-1, 1])  # [batch_size*n_video_step, 1]

                # alpha权重点乘上hidden 得上下文信息
                context = alpha * hidden1  # [batch_size*n_video_step, dim_hidden]

                # 把上下文变成3维
                context = tf.reshape(context, [self.batch_size, -1, self.dim_hidden])  # [batch_size, n_video_step, dim_hidden]

                # 在视频帧时间步这一维上求均值, 获得最终上下文
                context = tf.reduce_sum(context, axis=1)  # [batch_size, dim_hidden]


                # 当前词向量，上下文，和之前输出的拼接
                with tf.variable_scope('LSTM2'):
                    out2, state2 = self.lstm2(tf.concat(values=[current_embed, context, out_1], axis=1), state2)  # out2 -- [batch_size, dim_hidden]

                # caption取第i+1个标签，并扩宽1维
                labels = tf.expand_dims(caption[:, i+1], 1)  # [batch_size, 1]  新1扩出来的是第1维

                # 标记
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)  # [batch_size, 1]

                # 在第1维上对2者拼接,  把序号和label拼接
                concated = tf.concat([indices, labels], axis=1)  # [batch_size, 2]

                # 构造标签的onehot形式
                onehot_label = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)  # [batch_size, n_words]

                # 经过最后一层全连接，获得输出
                logit_words = tf.nn.xw_plus_b(out2, self.embed_word_W, self.embed_word_b)  # [batch_size, n_words]

                # 求交叉熵
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_label, logits=logit_words)  # [batch_size]

                # 交叉熵点乘掩码
                cross_entropy = cross_entropy * caption_mark[:, i]  # [batch_size]  # 非掩码的部分不是字，不重要

                # 保存每个logits
                probs.append(logit_words)

                # 算平均loss
                current_loss = tf.reduce_sum(cross_entropy) / self.batch_size

                # 累加算总loss
                loss += current_loss

        return loss, video, video_mark, caption, caption_mark, probs

    def build_generator(self):
        # 用保存的权重连接网络（网络有少许差别）
        # 每次之队一个视频，不用考虑batch_size考虑
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, dim_image])
        video_mark = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        # 图像转为嵌入
        video_flat = tf.reshape(video, [-1, self.dim_image])  # [n_video_lstm_step, dim_image]
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)  # [n_video_lstm_step, dim_hidden]
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])


        # 状态和lstm填充
        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        state3 = tf.zeros([1, self.lstm1.state_size])

        padding_lstm1 = tf.zeros([1, self.dim_hidden])
        padding_lstm2 = tf.zeros([1, 2*self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            list_out1 = []
            list_out3 = []
            list_state1 = []
            list_state3 = []
            for i in range(0, self.n_video_lstm_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope('LSTM1'):
                    out1, state1 = self.lstm1(image_emb[:, i, :], state1)  # 状态不断被下一个lstm使用  # out1--[batch_size, dim_hidden], state1--[batch_size, dim_hidden*2]
                    out3, state3 = self.lstm1(image_emb[:, n_video_lstm_step - i - 1, :], state3)
                    list_out1.append(out1)
                    list_out3.append(out3)
                    list_state1.append(state1)
                    list_state3.append(state3)

            for i in range(0, n_video_lstm_step):
                out_1 = tf.concat((list_out1[i], list_out3[n_video_lstm_step - i - 1]), axis=1)
                if i == 0:
                    hidden1 = out_1
                else:
                    hidden1 = tf.concat([hidden1, out_1], axis=0)

                with tf.variable_scope('LSTM2', reuse=tf.AUTO_REUSE):
                    out2, state2 = self.lstm2(tf.concat([padding_lstm2, out_1], axis=1), state2)  # out2=[1,dim_hidden] state2=[1, hidden*2]

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(0, self.n_caption_lstm_step):
                tf.get_variable_scope().reuse_variables()
                if i == 0:
                    with tf.device('/cpu:0'):
                        current_embbed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int32))  # 第一个就输入<eos> [1, dim_hidden]

                with tf.variable_scope('LSTM1'):
                    out1, state1 = self.lstm1(padding_lstm1, state1)
                    out3, state3 = self.lstm1(padding_lstm1, state3)
                    out_1 = tf.concat((out1, out3), axis=1)

                # hidden2 为编码部分， lstm2的输出out2
                hidden2 = tf.reshape(out2, [-1, 1])  # [hidden1, 1]

                # 得出hidden1对上一个隐向量的影响
                alpha = tf.matmul(hidden1, self.attention_W)  # [n_video_lstm_step, dim_hidden]

                # 根据据前一步的隐向量，得出对每个视频帧的配比
                alpha = tf.matmul(alpha, hidden2)  # [n_video_lstm_step, 1]
                #具体配比用softmax得
                alpha = tf.reshape(alpha, [1, -1])  # [1, n_video_lstm_step]
                alpha = tf.nn.softmax(alpha)  # [1, n_video_lstm_step]
                alpha = tf.reshape(alpha, [-1, 1])  # [n_video_lstm_step, 1]

                # 获得每个视频步提供的上下文, 点乘
                content = alpha * hidden1  # [n_video_lstm_step, dim_hidden]
                # 以其均值作为最后的上下文
                content = tf.reduce_sum(content, axis=0)  # [n_video_lstm_step]
                content = tf.reshape(content, [1, -1])  # [1, n_video_lstm_step]

                # 拼接上一个词嵌入，上下文，lstm1的输出
                with tf.variable_scope('LSTM2'):
                    out2, state2 = self.lstm2(tf.concat(values=[current_embbed, content, out_1], axis=1), state2)

                # 加一个全连接预测词
                logit_words = tf.nn.xw_plus_b(out2, self.embed_word_W, self.embed_word_b)

                # 获得最大可能性的预测标签
                max_prob_id = tf.arg_max(logit_words, 1)[0]

                # 保存输出id
                generated_words.append(max_prob_id)
                probs.append(logit_words)

                with tf.device('/cpu:0'):
                    current_embbed = tf.nn.embedding_lookup(self.Wemb, max_prob_id)  # [512] 因为1个拿出来的是1行，还要把单独的这一行填充到一个列表中
                    current_embbed = tf.expand_dims(current_embbed, axis=0)  # [1, 512]

                # 保存每次输出的词嵌入
                embeds.append(current_embbed)

        return video, video_mark, generated_words, probs, embeds


def get_video_data(data_file, feature_file):
    video_data = pd.read_csv(data_file, sep=',')

    video_data = video_data[video_data['Language'] == 'Chinese']  # 只获取英语标注的数据
    video_data['video_path'] = video_data.apply(lambda x: x['VideoID']+'_'+str(int(x['Start']))+'_'+str(int(x['End']))+'.avi.npy', axis=1)
    video_data['video_path'] = video_data['video_path'].map(lambda x: os.path.join(feature_file, x))
    video_data = video_data[video_data['video_path'].map(lambda x: os.path.exists(x))]
    video_data = video_data[video_data['Description'].map(lambda x: isinstance(x, str))]

    unifilename = sorted(video_data['video_path'].unique())
    video_data = video_data[video_data['video_path'].map(lambda x: x in unifilename)]  # x肯定在去重后的名单里啊，这句话是有什么作用

    return video_data


def get_word_vec_dict():
    model_text_path = '../word2vec/word_vec_video_ch.txt'
    model_file = open(model_text_path, 'r', encoding='utf8')
    i = -1
    words = 0
    vec_dim = 0
    word_vec_dict = {}

    for line in model_file:
        i += 1
        if i == 0:
            line_l = line.strip().split(' ')
            words = int(line_l[0])
            vec_dim = int(line_l[1])
            continue
        line_l = line.strip().split(' ')
        if i > words:
            break

        word_vec_dict[line_l[0]] = np.array(list(map(lambda x: float(x), line_l[1:])), dtype=np.float32)

    print('load word vector success')
    return word_vec_dict


def preProBuildWordVocab(sentence_iterator, word_count_threshold=1):
    # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))  # 最低词频为5
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        #compute word occurrence
        nsents += 1
        for w in sent.split(' '):
            if w == '':
                continue
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]  # 获取词汇表
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix = {}
    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for idx, w in enumerate(vocab):  # 4个特殊符号放在前面，单词放在后面
        wordtoix[w] = idx+4
        ixtoword[idx+4] = w

    # 使用预训练的word2vec向量
    word2vec_dict = get_word_vec_dict()
    word_vec_list = []
    for index in ixtoword:
        if ixtoword[index] in word2vec_dict:
            word_vec_list.append(word2vec_dict[ixtoword[index]])
        else:
            word_vec_list.append(np.random.uniform(-0.1, 0.1, (dim_hidden,)))
    word_vec_list = np.array(word_vec_list, dtype=np.float32)

    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents

    bias_init_vector = np.array([1.0 * word_counts[ ixtoword[i] ] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range  # 依据词频够建初始化bias

    return wordtoix, ixtoword, bias_init_vector, word_vec_list


def train(clean=False):
    train_data = get_video_data(video_train_data_path, video_feature_path)

    train_caption = train_data['Description'].values

    test_date = get_video_data(video_test_data_path, video_feature_path)

    test_caption = test_date['Description'].values

    caption_list = list(train_caption) + list(test_caption)

    captions = np.asarray(caption_list, np.object)
    captions = list(map(lambda x: x.replace('.', ''), captions))  # 去除一些标点符号
    captions = list(map(lambda x: x.replace(',', ''), captions))
    captions = list(map(lambda x: x.replace('"', ''), captions))
    captions = list(map(lambda x: x.replace('\n', ''), captions))
    captions = list(map(lambda x: x.replace('?', ''), captions))
    captions = list(map(lambda x: x.replace('!', ''), captions))
    captions = list(map(lambda x: x.replace('\\', ''), captions))
    captions = list(map(lambda x: x.replace('/', ''), captions))
    captions = list(map(lambda x: x.lower().strip(), captions))


    # wordtoix, ixtoword, bias_init_vector, word_vec_list = preProBuildWordVocab(captions,word_count_threshold=2)
    # np.save(wordtoix_path, wordtoix)
    # np.save(ixtoword_path, ixtoword)
    # np.save(bias_init_vector_path, bias_init_vector)

    if clean == True:

        wordtoix, ixtoword, bias_init_vector, word_vec_list = preProBuildWordVocab(captions)  # 获取词到id，id到词，以及向量初始化bias

        np.save(wordtoix_path, wordtoix)
        np.save(ixtoword_path, ixtoword)
        np.save(bias_init_vector_path, bias_init_vector)
        # np.save(word_vec_path, word_vec_list)
        model = Video_Caption_Generator(dim_image=dim_image,
                                        n_words=len(ixtoword),
                                        dim_hidden=dim_hidden,
                                        batch_size=batch_size,
                                        n_lstm_steps=n_frame_step,
                                        n_video_lstm_step=n_video_lstm_step,
                                        n_caption_lstm_step=n_caption_lstm_step,
                                        bias_init_vector=bias_init_vector,
                                        word_vec_list=word_vec_list
                                        )

    else:
        wordtoix = np.load(wordtoix_path).tolist()
        ixtoword = np.load(ixtoword_path)
        log_out.write(str(ixtoword.tolist()))
        # print(ixtoword.tolist())
        print('save id2word in text')
        bias_init_vector = np.load(bias_init_vector_path)

        #wordtoix, ixtoword, bias_init_vector, word_vec_list = preProBuildWordVocab(captions)  # 获取词到id，id到词，以及向量初始化bias
        model = Video_Caption_Generator(dim_image=dim_image,
                                n_words=len(wordtoix),
                                dim_hidden=dim_hidden,
                                batch_size=batch_size,
                                n_lstm_steps=n_frame_step,
                                n_video_lstm_step=n_video_lstm_step,
                                n_caption_lstm_step=n_caption_lstm_step,
                                bias_init_vector=bias_init_vector,

                                )

    # 建模型， 并把输入输出引出来
    tf_loss, tf_video, tf_video_mark, tf_caption, tf_caption_mark, tf_probs = model.build_model()

    # 记录loss
    loss_fd = open(loss_out_path, 'a')
    if not os.path.exists(loss_save_path):
        loss_to_draw = []
    else:
        loss_to_draw = list(np.load(loss_save_path))

    # 记录epoch
    if not os.path.exists(epoch_save_path):
        epoch_now = [-1]
    else:
        epoch_now = list(np.load(epoch_save_path))

    # 开始启动session
    sess = tf.InteractiveSession()

    # 模型保存管理
    saver = tf.train.Saver()
    # 构造需要运行的tensor
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)

    # 初始化网络
    sess.run(tf.global_variables_initializer())
    # 模型导入之前要初始化变量后
    if not clean:
        # 第一次还没有模型的时候
        if not os.path.exists(save_models_path):
            os.mkdir(save_models_path)

        else:
            # 只有目录没模型
            if not os.path.exists(save_models_path+'checkpoint'):
                pass
            else:
                saver.restore(sess, tf.train.latest_checkpoint(save_models_path))



    begin_epoch = epoch_now[0] + 1

    # 开始每一步运行
    for epoch in range(begin_epoch, n_epochs):
        # 每个epoch要画的loss
        loss_to_draw_epoch = []

        # 获取标签
        index = list(train_data.index)
        # 打乱标签
        np.random.shuffle(index)
        # 打乱数据
        train_data = train_data.ix[index]

        # 根据路径获取分组数据
        train_data_group = train_data.groupby('video_path')

        # 一个视频只随机获取一个标签
        #current_train_data = train_data_group.apply(lambda x: x.iloc[np.random.choice(len(x))])

        # 一个视频获取第一个标签
        current_train_data = train_data_group.apply(lambda x: x.iloc[0])

        # 移除原来的index
        current_train_data = current_train_data.reset_index(drop=True)

        # 取数据， 每次取个batch_size
        for start, end in list(zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data), batch_size)
        )):
            start_time = time.time()
            # 获取当前batch
            current_batch = current_train_data[start:end]
            # 获取视频特征路径
            current_videos = current_batch['video_path'].values
            # 存放视频特征的向量 . 数据输入前用np构造
            current_feature = np.zeros((batch_size, n_video_lstm_step, dim_image))
            # 提取视频特征向量
            current_feature_val = list(map(lambda x: np.load(x), current_videos))  # [batch_size, n_video_lstm_step, dim_image]
            # 特征掩码
            current_feature_mark = np.zeros((batch_size, n_video_lstm_step))   # [batch_size, n_video_lstm_step]

            # 依次填充特征向量
            for idx, feature in enumerate(current_feature_val):
                current_feature[idx][:len(feature)] = feature  # feature-[n_video_lstm_step, dim_image]
                current_feature_mark[idx][:len(feature)] = 1

            # 获取当前batch_size标签
            current_captions = current_batch['Description'].values
            current_captions = list(map(lambda x: '<bos> ' + x, current_captions))
            current_captions = list(map(lambda x: x.replace('.', ''), current_captions))
            current_captions = list(map(lambda x: x.replace(',', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('"', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\n', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('?', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('!', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('\\', ''), current_captions))
            current_captions = list(map(lambda x: x.replace('/', ''), current_captions))
            current_captions = list(map(lambda x: x.lower().strip(), current_captions))


            # 构造<bos>
            for idx, each_capt in enumerate(current_captions):
                word_list = each_capt.split(' ')
                # 小于最大文本长度， 直接在后面加<bos>
                if len((each_capt)) < n_caption_lstm_step:
                    current_captions[idx] = each_capt + ' <eos>'
                else:
                    new_capt = ''
                    for word in word_list:
                        new_capt += word + ' '
                    current_captions[idx] = new_capt + '<eos>'

            #print(current_captions)
            log_out.write(str(current_captions))
            # 化成id列表
            current_caption_id_list = []
            for each_capt in current_captions:
                current_word_id_list = []
                word_list = each_capt.split()
                for word in word_list:
                    if word in wordtoix:
                        current_word_id_list.append(wordtoix[word])
                    else:
                        current_word_id_list.append(wordtoix['<unk>'])
                current_caption_id_list.append(current_word_id_list)

            #print(current_caption_id_list)
            log_out.write(str(current_caption_id_list))
            # 获取caption矩阵
            # 不足的用0填充
            current_caption_matrix = sequence.pad_sequences(current_caption_id_list,  padding='post', maxlen=n_caption_lstm_step)  # [batch_size, n_caption_lstm_step]_
            # 在列（第1维）扩宽了1, 并以0填充, 并化为int型  因为第一个是bos, 所以标签往后挪一个
            current_caption_matrix = np.hstack((current_caption_matrix, np.zeros([len(current_caption_matrix), 1]))).astype(int)  # [batch_size, n_caption_lstm_step]_
            # caption矩阵掩码
            current_caption_mark = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))

            # 找出非0个数，加1之后就是第一个非0数的后1个  ？？caption都往后取1个
            nozeros = np.array(list(map(lambda x: (x != 0).sum()+1, current_caption_matrix)))

            # 构造掩码，非0的为1
            for idx, row in enumerate(current_caption_mark):
                current_caption_mark[idx][:nozeros[idx]] = 1

            # 输入数据算预测
            probs_val = sess.run(tf_probs, feed_dict={
                tf_video: current_feature,
                tf_caption: current_caption_matrix})

            # 最小化求loss
            _, loss_val = sess.run([train_op, tf_loss],
                                   feed_dict={
                                       tf_video: current_feature,
                                       tf_video_mark: current_feature_mark,
                                       tf_caption: current_caption_matrix,
                                       tf_caption_mark: current_caption_mark
                                   })

            #  保存loss
            loss_to_draw_epoch.append(loss_val)
            print('idx:', start, ' epoch:', epoch, ' loss:', loss_val, ' time:', str((time.time()-start_time)))
            loss_fd.write('idx:'+str(start)+' epoch:'+str(epoch)+' loss:'+str(loss_val)+' time:'+str((time.time()-start_time))+'\n')


        # 每个epoch画一个图
        loss_to_draw.append(np.mean(loss_to_draw_epoch))



        # 绘制loss图片
        plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
        plt.grid(True)  # 显示网格
        plt.savefig(loss_img_path)

        # 保存模型
        if (epoch+1) % save_epoch == 0:
            print('Epoch:'+ str(epoch) +'  Saving model')
            saver.save(sess, os.path.join(save_models_path,'model.ckpt'), global_step=epoch)
            np.save(loss_save_path, loss_to_draw)  # 把loss保存起来
            np.save(epoch_save_path, [epoch])  # 保存epoch
        if (epoch+1) % 50 == 0:
            plt.plot(range(len(loss_to_draw)), loss_to_draw, color='g')
            plt.grid(True)  # 显示网格
            save_loss_img_path = save_path + str(epoch)+ '_loss_img.png'
            plt.savefig(save_loss_img_path)

    loss_fd.close()


# save_path + 'models\\model.ckpt-457'
def test(model_path=None):
    test_data = get_video_data(video_test_data_path, video_feature_path)
    test_video = test_data['video_path'].unique()
    ixtoword = pd.Series(np.load(ixtoword_path).tolist())  # 为了后面可以整句话转
    bias_init_vector = np.load(bias_init_vector_path)

    model = Video_Caption_Generator(
        dim_image=dim_image,
        n_words=len(ixtoword),
        dim_hidden=dim_hidden,
        batch_size=batch_size,
        n_lstm_steps=n_frame_step,
        n_video_lstm_step=n_video_lstm_step,
        n_caption_lstm_step=n_caption_lstm_step,
        bias_init_vector=bias_init_vector
    )

    video_tf, video_mark_tf, generate_word_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    if not model_path:
        saver.restore(sess, tf.train.latest_checkpoint(save_models_path))
    else:
        saver.restore(sess, model_path)

    text_out = open('../save/S2VT_out2.txt', 'a')
    for idex, video_feat_path in enumerate(test_video):
        print(idex, video_feat_path)
        video_feat = np.load(video_feat_path)[None,...]
        if video_feat.shape[1] == n_frame_step:
            video_mark = np.ones([video_feat.shape[0], video_feat.shape[1]])
        else:
            continue
        generate_word_idx = sess.run(generate_word_tf, feed_dict={
            video_tf:video_feat,
            video_mark_tf:video_mark
        })

        generate_word = ixtoword[generate_word_idx]  # 用了pandas， 所以可以取出全部
        # 得出等于eos的那一项的后面一个
        punctuation = np.argmax(np.array(generate_word)=='<eos>') + 1
        # 截取有用的部分
        generate_word = generate_word[:punctuation]

        # 去掉bos,eos
        generate_sent = ' '.join(generate_word)
        generate_sent = generate_sent.replace('<bos> ', '')
        generate_sent = generate_sent.replace(' <eos>', '')
        print(generate_sent)
        text_out.write(video_feat_path + '\n')
        text_out.write(generate_sent+'\n\n')


if __name__ == '__main__':
#   train(clean=True)
    test('E:\1我的东西\我的很多资料\研究生\python project\VideoCaptioning_att_czy\save_1101\models\model.ckpt-10.meta')
