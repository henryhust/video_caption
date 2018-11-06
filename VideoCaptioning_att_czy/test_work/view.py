import cv2
import numpy as np
import skimage
import tensorflow as tf
import pandas as pd
import os
from test_work import test_model_RGB as model_RGB
seg_time = 8

save_path = '..\\save\\'
if not os.path.exists(save_path+'processing_video\\'):
    os.mkdir(save_path+'processing_video\\')

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

def extract_video_features(video_path_list, num_frames = 80, vgg16_model='..\\vgg16-20160129.tfmodel'):
    print('Loading VGG16 model..')
    with open(vgg16_model, mode='rb') as f:
      fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })
    graph = tf.get_default_graph()

    video_feat_list = []

    for video_path in video_path_list:
        print("Extracting video features for: " + os.path.basename(video_path))
    # Load tensorflow VGG16 model and setup computation graph


        # Read video file
        try:
            cap = cv2.VideoCapture(video_path)
        except:
            pass

        #extract frames from video
        frame_count = 0
        frame_list = []

        while True:
            #extract frames from the video, where each frame is an array (height*width*3)
            ret, frame = cap.read()
            if ret is False:
                break
            frame_list.append(frame)
            frame_count += 1
        frame_list = np.array(frame_list)

        # select num_frames from frame_list if frame_cout > num_frames
        if frame_count > num_frames:
            frame_indices = np.linspace(0, frame_count, num=num_frames, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        # crop/resize each frame
        #cropped_frame_list is a list of frames, where each frame is a height*width*3 ndarray
        cropped_frame_list = np.asarray(list(map(lambda x: preprocess_frame(x), frame_list)))

        # extract fc7 features from VGG16 model for each frame
        # feats.shape = (num_frames, 4096)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
            video_feat = sess.run(fc7_tensor, feed_dict={images: cropped_frame_list})
            video_feat_list.append(video_feat)


    return video_feat_list

def get_caption(video_feat_list, model_path=save_path + 'models\\model.ckpt-627'):
    print("Generating caption ...")
    #video_feat_path = os.path.join('./temp_RGB_feats', '8e0yXMa708Y_24_33.avi.npy')
    ixtoword = pd.Series(np.load(save_path +'\\ixtoword.npy').tolist()) #, encoding='latin1'
    bias_init_vector = np.load(save_path +'\\bias_init_vector.npy')

    # lstm parameters
    dim_image = 4096
    dim_hidden= 512
    n_video_lstm_step = 80
    n_caption_lstm_step = 20
    n_frame_step = 80
    batch_size = 20

    #setup lstm encoder-decoer with attention model
    model = model_RGB.Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    #restore lstm model parameters
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    generated_sentence_list = []
    for video_feat in video_feat_list:
        video_feat = video_feat[None,...]

        if video_feat.shape[1] == n_frame_step:
            video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        # run model and obatin the embeded words (indices)
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

        # convert indices to words
        generated_words = ixtoword[generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
        generated_words = generated_words[:punctuation]
        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('<bos> ', '')
        generated_sentence = generated_sentence.replace(' <eos>', '')
        print(generated_sentence,'\n')
        generated_sentence_list.append(generated_sentence)

    return generated_sentence_list

def get_video_feats(feat_path):
    current_videos = os.listdir(feat_path)
    current_feature_val = list(map(lambda x: np.load(feat_path + x), current_videos))
    return current_feature_val

def get_video_segment(src_video_path):

    video_segment_list = []
    videoCapture = cv2.VideoCapture(src_video_path)  # 从文件读取视频

    i = 1
    j = 1

    # 判断视频是否打开
    if (videoCapture.isOpened()):
        print('Open')
    else:
        print('Fail to open!')

    fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率

    size = (
    int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # 获取原视频帧的大小
    now_video_path = save_path + 'processing_video\\' + str(j) + '.avi'
    video_segment_list.append(now_video_path)
    videoWriter = cv2.VideoWriter(now_video_path,
                                  cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)  # 新视频保存路径和参数

    success, frame = videoCapture.read()  # 读取第一帧

    time_long = int(fps * seg_time)  # 每次截8秒

    while success:
        videoWriter.write(frame)  # 写入“新视频”
        i = i + 1
        #print(str(i))

        if (i % time_long == 0):  # 截取相应长达的视频
            j = j + 1
            now_video_path = save_path + 'processing_video\\' + str(j) + '.avi'
            video_segment_list.append(now_video_path)
            videoWriter = cv2.VideoWriter(now_video_path,
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)  # 新视频保存路径和参数

        success, frame = videoCapture.read()  # 循环读取下一帧
    return video_segment_list[:-1]

def write_video_captions(generated_sentence_list, out_file):
    out = open(out_file, 'w')
    for i in range(len(generated_sentence_list)):
        time_label = str(i*seg_time) + '-' + str((i+1)*seg_time) + ' s : '
        out.write(time_label+generated_sentence_list[i]+'\n\n')
    out.close()


if __name__ == '__main__':
    #video_path='/home/ytkj/czy/VideoCaptioning_att-master/data/MSVD/YouTubeClips/_0nX-El-ySo_83_93.avi'
    #video_path='D:\\file\\待整理\\毕设\\ThreePastShop1front.mpg'
    video_path = 'D:\\file\\待整理\\毕设\\VideoCaptioning_att-master_czy\\test_supervisory_RGB_feats\\'
    # video_path_list = get_video_segment(video_path)
    # video_feat_list = extract_video_features(video_path_list)
    video_feat_list = get_video_feats(video_path)
    generated_sentence_list = get_caption(video_feat_list)
    write_video_captions(generated_sentence_list, save_path + 'processing_video\\video_captures_1000.text')
