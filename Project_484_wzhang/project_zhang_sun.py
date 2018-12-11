import time
import numpy as np
import tensorflow as tf
import pprint
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt
from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage

times=1
rate=1e-4
photo_size=33
tag_size=21
pick=128
dim=1
weight=3
stride=14
cp_dir="checkpoint"
result_addr="result"
sess = tf.Session()

# ---------------------------------- Building model ----------------------------------
photo_holder = tf.placeholder(tf.float32, [None, photo_size, photo_size, dim], name='photo')
tag_holder = tf.placeholder(tf.float32, [None, tag_size, tag_size, dim], name='tags')

degree = {
    'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
    'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
    'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
}
noise = {
    'b1': tf.Variable(tf.zeros([64]), name='b1'),
    'b2': tf.Variable(tf.zeros([32]), name='b2'),
    'b3': tf.Variable(tf.zeros([1]), name='b3')
}

conv1 = tf.nn.relu(tf.nn.conv2d(photo_holder, degree['w1'], strides=[1,1,1,1], padding='VALID') + noise['b1'])
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, degree['w2'], strides=[1,1,1,1], padding='VALID') + noise['b2'])
pred = tf.nn.conv2d(conv2, degree['w3'], strides=[1,1,1,1], padding='VALID') + noise['b3']

# Loss function (MSE)
loss = tf.reduce_mean(tf.square(tag_holder - pred))
saver = tf.train.Saver()

def calculate(sub_photo, weight=3):
    if len(sub_photo.shape) == 3:
        row, col, _ = sub_photo.shape
        row = row - np.mod(row, weight)
        col = col - np.mod(col, weight)
        sub_photo = sub_photo[0:row, 0:col, :]
    else:
        row, col = sub_photo.shape
        row = row - np.mod(row, weight)
        col = col - np.mod(col, weight)
        sub_photo = sub_photo[0:row, 0:col]
    return sub_photo

def pre_processing(path, weight=3):
  pre_photo = scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  pre_tag = calculate(pre_photo, weight)
  pre_photo = pre_photo / 255.
  pre_tag = pre_tag / 255.
  pre_result = scipy.ndimage.interpolation.zoom(pre_tag, (1./weight), prefilter=False)
  pre_result = scipy.ndimage.interpolation.zoom(pre_result, (weight/1.), prefilter=False)

  return pre_result, pre_tag

def read_checkpoint(cp_dir):
    # print("Reading checkpoints...")
    m_addr = "%s_%s" % ("srcnn", tag_size)
    cp_dir = os.path.join(cp_dir, m_addr)

    point = tf.train.get_checkpoint_state(cp_dir)
    if point and point.model_checkpoint_path:
        ckpt_name = os.path.basename(point.model_checkpoint_path)
        saver.restore(sess, os.path.join(cp_dir, ckpt_name))
        return True
    else:
        return False

def train_initialization():
    # Preparing data
    train_data_set="Train"
    filenames = os.listdir(train_data_set)
    data_addr = os.path.join(os.getcwd(), train_data_set)
    data = glob.glob(os.path.join(data_addr, "*.bmp"))

    data_sequence = []
    tag_sequence = []
    tire = abs(photo_size - tag_size) / 2 # 6
    for i in range(len(data)):
        # Preprocessing
        pre_input, pre_tags = pre_processing(data[i], weight)

        if len(pre_input.shape) == 3:
            row, col, _ = pre_input.shape
        else:
            row, col = pre_input.shape

        for x in range(0, row-photo_size+1, stride):
            for y in range(0, col-photo_size+1, stride):
                re_data = pre_input[x:x+photo_size, y:y+photo_size] # [33 x 33]
                re_tag = pre_tags[x+int(tire):x+int(tire)+tag_size, y+int(tire):y+int(tire)+tag_size] # [21 x 21]
                re_data = re_data.reshape([photo_size, photo_size, 1])  
                re_tag = re_tag.reshape([tag_size, tag_size, 1])
                data_sequence.append(re_data)
                tag_sequence.append(re_tag)

    # Making data
    with h5py.File(os.path.join(os.getcwd(), 'checkpoint/train.h5'), 'w') as editor:
        editor.create_dataset('data', data=np.asarray(data_sequence))
        editor.create_dataset('tag', data=np.asarray(tag_sequence))

def test_initialization():
  # Preparing data
    test_data_set="Test"
    data_addr = os.path.join(os.sep, (os.path.join(os.getcwd(), test_data_set)), "bmp")
    data = glob.glob(os.path.join(data_addr, "*.bmp"))

    data_sequence = []
    tag_sequence = []
    tire = abs(photo_size - tag_size) / 2 # 6

    # Preprocessing
    input_, label_ = pre_processing(data[2], weight)

    if len(input_.shape) == 3:
        row, col, _ = input_.shape
    else:
        row, col = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    vec1 = vec2 = 0 
    for x in range(0, row-photo_size+1, stride):
        vec1 += 1; vec2 = 0
        for y in range(0, col-photo_size+1, stride):
            vec2 += 1
            re_data = input_[x:x+photo_size, y:y+photo_size] # [33 x 33]
            re_tag = label_[x+int(tire):x+int(tire)+tag_size, y+int(tire):y+int(tire)+tag_size] # [21 x 21]
            re_data = re_data.reshape([photo_size, photo_size, 1])  
            re_tag = re_tag.reshape([tag_size, tag_size, 1])
            data_sequence.append(re_data)
            tag_sequence.append(re_tag)

    # Making data
    with h5py.File(os.path.join(os.getcwd(), 'checkpoint/test.h5'), 'w') as editor:
        editor.create_dataset('data', data=np.asarray(data_sequence))
        editor.create_dataset('tag', data=np.asarray(tag_sequence))

    return vec1, vec2

def train(sess, cp_dir):
    train_initialization()
    # read_checkpoint(cp_dir)
    buff = 0
    start_time = time.time()
    data_dir = os.path.join('./{}'.format(cp_dir), "train.h5")
    
    # Reading Data
    with h5py.File(data_dir, 'r') as editor:
        train_data_set = np.array(editor.get('data'))
        train_tag_set = np.array(editor.get('tag'))

    # Stochastic gradient descent with the standard backpropagation
    train_gradient_descent = tf.train.GradientDescentOptimizer(rate).minimize(loss)
    tf.initialize_all_variables().run()
    read_checkpoint(cp_dir)
    print("Training...")
    for ep in range(times):
        # Run by batch images
        batch_idxs = len(train_data_set) // pick
        for idx in range(0, batch_idxs):        
            train_pick = train_data_set[idx*pick : (idx+1)*pick]
            train_tags = train_tag_set[idx*pick : (idx+1)*pick]

            buff += 1
            _, err = sess.run([train_gradient_descent, loss], feed_dict={photo_holder: train_pick, tag_holder: train_tags})

            if buff % 10 == 0:
                print("%d\tstep: %d\t time: %f\t loss: %f" % ((ep+1), buff, time.time()-start_time, err))

            if buff % 500 == 0:
                model_name = "SRCNN.model"
                model_dir = "%s_%s" % ("srcnn", tag_size)
                cp_dir = os.path.join(cp_dir, model_dir)
                if not os.path.exists(cp_dir):
                    os.makedirs(cp_dir)
                saver.save(sess, os.path.join(cp_dir, model_name), global_step=buff)

def test(sess, cp_dir) :
    v1, v2 = test_initialization()
    read_checkpoint(cp_dir)
    data_dir = os.path.join('./{}'.format(cp_dir), "test.h5")
    
    # Reading Data
    with h5py.File(data_dir, 'r') as hf:
        train_data_set = np.array(hf.get('data'))
        train_tag_set = np.array(hf.get('tag'))

    train_op = tf.train.GradientDescentOptimizer(rate).minimize(loss)
    tf.initialize_all_variables().run()

    print("Testing...")
    result = pred.eval({photo_holder: train_data_set, tag_holder: train_tag_set})
    # Merging
    size = [v1, v2]
    row, col = result.shape[1], result.shape[2]
    photo_matrix = np.zeros((row*size[0], col*size[1], 1))
    for idx, image in enumerate(result):
        i = idx % size[1]
        j = idx // size[1]
        photo_matrix[j*row:j*row+row, i*col:i*col+col, :] = image
    result = photo_matrix
    result = result.squeeze()
    photo_address = os.path.join(os.getcwd(), result_addr)
    photo_address = os.path.join(photo_address, "test_image.png")
    scipy.misc.imsave(photo_address, result)
 

# -------------------------------------- Main --------------------------------------
def main(_):

    if not os.path.exists(cp_dir):
        os.makedirs(cp_dir)
    if not os.path.exists(result_addr):
        os.makedirs(result_addr)

    with tf.Session() as sess:
        train(sess, cp_dir)
        test(sess, cp_dir)

if __name__ == '__main__':
    tf.app.run()
