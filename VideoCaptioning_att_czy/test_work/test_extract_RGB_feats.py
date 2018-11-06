import os
import tensorflow as tf
import cv2
import numpy as np
import skimage

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1,3"

# 对图片进行预处理
def preprocess_frame(image, target_height=224, target_width=224):
    # 图像为2维， 第0维copy 3次
    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    # 图像为4维， 最后一维只拿第0项
    if len(image.shape) == 4:
        image = image[:,:,:,0]

    # 将图像有unit8转float
    image = skimage.img_as_float(image).astype(np.float32)  # [360, 480, 3]
    # 获取各个维度
    height, width, rgb = image.shape
    # 大小等长，用缩小的
    if height == width:
        # cv2 是先width再height, 为什么这里之后就没一rgb这维
        resized_image = cv2.resize(image, (target_width, target_height))
    # 高度小于宽度，以高度为准
    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height),target_height))  # [224, 298, 3]
        crop_length = int((resized_image.shape[1] - target_width) / 2)  # 37
        resized_image = resized_image[:, crop_length:resized_image.shape[1]-crop_length]  # [224, 224, 3]
    else:
        resized_image = cv2.resize(image, (target_width, int(height * float(target_width) / width)))
        crop_length = int((resized_image.shape[0] - target_height) / 2)
        resized_image = resized_image[crop_length:resized_image.shape[0]-crop_length, :]

    return cv2.resize(resized_image, (target_width, target_height))

def main():
    num_frames = 80
    video_path = '/home/ytkj/czy/VideoCaptioning_att-master/data/MSVD/YouTubeClips'
    video_save_path = '../save/temp_RGB_feats'
    videos = os.listdir(video_path)
    # 过滤掉非视频文件
    videos = filter(lambda x: x.endswith('.avi'), videos)

    # 载入tensorflow模型
    with open('../vgg16-20160129.tfmodel', mode='rb') as f:
        file_content = f.read()

    # 把模型以string的方式导入预定义的图中
    graphDef = tf.GraphDef()
    graphDef.ParseFromString(file_content)

    # 构造输入映射
    images = tf.placeholder("float", [None, 224, 224, 3])
    tf.import_graph_def(graphDef, input_map={'images': images})
    # 获取预定义的图
    graph = tf.get_default_graph()

    # 依次处理每个视频
    for idx, now_video_path in enumerate(videos):
        print(idx, now_video_path)
        if os.path.exists(os.path.join(video_save_path, now_video_path)):
            print('Already processed ...')
            continue

        video_full_path = os.path.join(video_path, now_video_path)
        try:
            # 用opencv读取视频
            cap = cv2.VideoCapture(video_full_path)
        except:
            print('error in cv2')

        # 帧计数
        frame_count = 0
        # 帧保存
        frame_list = []

        # 读取所有视频帧
        while True:
            ret, frame = cap.read()         # 读取了多少帧怎么确定的
            if ret is False:
                break
            frame_list.append(frame)
            frame_count += 1

        # 将帧列表变成numpy形式
        frame_list = np.array(frame_list)

        # 如果大于80帧，均匀选取80帧
        if frame_count > 80:
            frame_indices = np.linspace(0, frame_count, num=80, endpoint=False).astype(int)
            frame_list = frame_list[frame_indices]

        # 依次对每个图片进行预处理，剪裁
        crop_frame_list = np.asarray(list(map(lambda x:preprocess_frame(x), frame_list)))

        # 开始运行session
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            fc7_tensor = graph.get_tensor_by_name('import/Relu_1:0')
            # run出featurelai
            feats = sess.run(fc7_tensor,feed_dict={images: crop_frame_list})
        save_full_path = os.path.join(video_save_path, now_video_path+'.npy')

        np.save(save_full_path, feats)
        print(feats.shape)

if __name__ == '__main__':
    main()


