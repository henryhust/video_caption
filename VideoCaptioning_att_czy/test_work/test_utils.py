import numpy as np
import skimage
import tensorflow as tf
import pandas as pd
import os
import test_work.test_model_RGB as model_RGB
import cv2

def preprocess_frame(image, target_height=224, target_width=224):
    #function to resize frames then crop
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    image = skimage.img_as_float(image).astype(np.float32)
    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_width,target_height))

    elif height < width:
        #cv2.resize(src, dim) , where dim=(width, height)
        #image.shape[0] returns height, image.shape[1] returns width, image.shape[2] reutrns 3 (3 RGB channels)
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_height))
        cropping_length = int((resized_image.shape[1] - target_width) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_width, target_height))

def extract_video_feature(video_path, num_frame=80, vgg16_model_path='/home/ytkj/czy/VideoCaptioning_att-master/vgg16-20160129.tfmodel'):
    print('extract video feature for:', video_path)

    with open(vgg16_model_path, 'rb') as file:
        file_content = file.read()
    grapDef = tf.GraphDef()
    grapDef.ParseFromString(file_content)
    images = tf.placeholder('float', [None,224,224,3])
    tf.import_graph_def(grapDef, input_map={'images':images})
    grap = tf.get_default_graph()


    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('error in cv2')

    frame_count = 0
    frame_list = []

    while True:

        ret, frame = cap.read()
        if ret is False:
            break
        frame_list.append(frame)
        frame_count += 1
    frame_list = np.array(frame_list)

    if frame_count > num_frame:
        frame_indices = np.linspace(0, frame_count, num=num_frame, endpoint=False).astype(int)
        frame_list = frame_list[frame_indices]

    crop_frame_list = np.asarray(list(map(lambda x:preprocess_frame(x), frame_list)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        fc7_tensor = grap.get_tensor_by_name('import/Relu_1:0')
        video_feat = sess.run(fc7_tensor, feed_dict={images: crop_frame_list})

    return video_feat

def get_caption(video_feature, model_path='../save/models/model.ckpt-45'):
    print('generating caption ...')
    idtoword = pd.Series(np.load('../save/ixtoword.npy').tolist())
    bias_init_vector = np.load('../save/bias_init_vector.npy')

    dim_image = 4096
    dim_hidden = 512
    n_lstm_video_step = 80
    n_lstm_caption_step = 20
    n_frame_step = 80

    batch_size= 5
    model = model_RGB.Video_Caption_Generator(dim_image,
                                              len(idtoword),
                                              dim_hidden,
                                              batch_size,
                                              n_frame_step,
                                              n_lstm_video_step,
                                              n_lstm_caption_step,
                                              bias_init_vector)

    video_tf, video_mark_tf, generated_word_tf, probs_tf, last_word_emb_tf = model.build_generator()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    video_feature = video_feature[None,...]
    if video_feature.shape[1] == n_frame_step:
        video_mark = np.ones((video_feature.shape[0], video_feature.shape[1]))

    generated_word_id = sess.run(generated_word_tf, feed_dict={video_tf: video_feature, video_mark_tf:video_mark})

    generated_words = idtoword[generated_word_id]

    punc = np.argmax(np.array(generated_words) == '<eos>') + 1

    generated_words = generated_words[:punc]

    generated_sent = ' '.join(generated_words)
    generated_sent = generated_sent.replace('<bos> ', '').replace(' <eos>', '')

    print(generated_sent, '\n')
    return generated_sent

if __name__ == '__main__':

    video_path = '/home/ytkj/视频/twopeoplewalk.avi'
    video_feat = extract_video_feature(video_path)
    get_caption(video_feat)
