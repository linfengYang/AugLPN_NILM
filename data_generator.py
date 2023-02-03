from Logger import log
import numpy as np
import pandas as pd


import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, mains, appliances_regression, appliances_classification, window_size, batch_size, shuffle=False):
        self.mains = mains
        self.appliances_regression = appliances_regression
        self.appliances_classification = appliances_classification
        self.window_size = window_size
        self.batch_size = batch_size
        self.indices = np.arange(len(self.mains) - self.window_size + 1)  # 可以往下滑多少次窗口
        self.shuffle = shuffle
        self.total_size = 0

    def __len__(self):
        return np.ceil(len(self.indices) / self.batch_size)

    def feed_chunk(self):

        max_batchsize = len(self.mains) - self.window_size + 1
        if self.batch_size < 0:
            self.batch_size = max_batchsize

        # define indices
        indices = np.arange(max_batchsize)

        # providing sliding windows:
        i = 0
        for start_idx in range(0, max_batchsize, self.batch_size):
            excerpt = indices[start_idx:start_idx + self.batch_size]

            mains_batch_np = np.array([self.mains[idx:idx + self.window_size] for idx in excerpt])
            appliances_regression_batch_np=np.array([self.appliances_regression[idx:idx + self.window_size] for idx in excerpt])
            appliances_classification_batch_np = np.array([self.appliances_classification[idx:idx + self.window_size] for idx in excerpt])


            #  why reshape?
            # mains_batch_np = np.reshape(mains_batch_np, (mains_batch_np.shape[0], mains_batch_np.shape[1], 1))
            # appliances_regression_batch_np = np.reshape(appliances_regression_batch_np,
            #                                             (appliances_regression_batch_np.shape[0],
            #                                              appliances_regression_batch_np.shape[1]))
            # appliances_classification_batch_np = np.reshape(appliances_classification_batch_np,
            #                                                 (appliances_classification_batch_np.shape[0],
            #                                                  appliances_classification_batch_np.shape[1]))
            # print('mains_batch_np:',mains_batch_np.shape)  # 128*512
            # print('appliances_regression_batch_np:',appliances_regression_batch_np.shape)
            # print('appliances_classification_batch_np:', appliances_classification_batch_np.shape)
            yield mains_batch_np, [appliances_regression_batch_np, appliances_classification_batch_np]

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indices)


class TestDataGenerator(object):

    def __init__(self, nofWindows, windowlength):

        self.nofWindows = nofWindows
        self.windowlength = windowlength

    def feed(self, inputs):

        inputs = inputs.flatten()
        # max_nofw = inputs.size - self.windowlength
        max_nofw = inputs.size - self.windowlength + 1
        print('max_nofw:', max_nofw)
        if self.nofWindows < 0:
            self.nofWindows = max_nofw

        indices = np.arange(max_nofw, dtype=int)

        # providing sliding windows:
        for start_idx in range(0, max_nofw, self.nofWindows):
            excerpt = indices[start_idx:start_idx + self.nofWindows]

            yield np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])
            #tar = np.array([inputs[idx:idx + self.windowlength] for idx in excerpt])





#
# class ChunkS2S_Slider(object):
#     def __init__(self, filename, batchsize, chunksize, shuffle, length, crop=None, header=0, ram_threshold=5 * 10 ** 5):
#
#         self.filename = filename  #training_path
#         self.batchsize = batchsize  #default=1000
#         self.chunksize = chunksize  #5*10**6
#         self.shuffle = shuffle  #True False
#         self.length = length  #windowlength=599
#         self.header = header
#         self.crop = crop
#         self.ram = ram_threshold
#         self.total_size = 0
#
#     def check_length(self):
#         # check the csv size
#         check_cvs = pd.read_csv(self.filename,
#                                 nrows=self.crop,
#                                 chunksize=10 ** 3,
#                                 header=self.header
#                                 )
#
#         for chunk in check_cvs:
#             size = chunk.shape[0]  # shape[0]为矩阵的行数
#             self.total_size += size
#             del chunk
#         log('Size of the dataset is {:.3f} M rows.'.format(self.total_size / 10 ** 6))
#         if self.total_size > self.ram:  # IF dataset is too large for memory
#             log('It is too large to fit in memory so it will be loaded in chunkes of size {:}.'.format(self.chunksize))
#         else:
#             log('This size can fit the memory so it will load entirely')
#
#     def feed_chunk(self):
#
#         if self.total_size == 0:
#             ChunkS2S_Slider.check_length(self)
#
#         if self.total_size > self.ram:  # IF dataset is too large for memory
#
#             # LOAD data from csv
#             data_frame = pd.read_csv(self.filename,
#                                      nrows=self.crop,
#                                      chunksize=self.chunksize,
#                                      header=self.header
#                                      )
#
#             skip_idx = np.arange(self.total_size/self.chunksize)
#             if self.shuffle:
#                 np.random.shuffle(skip_idx)
#
#             log(str(skip_idx), 'debug')
#
#             for i in skip_idx:
#
#                 log('index: ' + str(i), 'debug')
#
#                 # Read the data
#                 data = pd.read_csv(self.filename,
#                                    nrows=self.chunksize,
#                                    skiprows=int(i)*self.chunksize,
#                                    header=self.header)
#
#                 np_array = np.array(data)
#                 inputs, targets = np_array[:, 0], np_array[:, 1]
#
#                 # 对分类、回归两个子网络 分别进行数据处理
#
#
#
#
#                 max_batchsize = inputs.size - self.length + 1
#                 if self.batchsize < 0:
#                     self.batchsize = max_batchsize
#
#                 # define indices and shuffle them if necessary
#                 indices = np.arange(max_batchsize)
#                 if self.shuffle:
#                     np.random.shuffle(indices)
#
#                 # providing sliding windows:
#                 i = 0
#                 for start_idx in range(0, max_batchsize, self.batchsize):
#                     excerpt = indices[start_idx:start_idx + self.batchsize]
#
#                     inp = np.array([inputs[idx:idx + self.length] for idx in excerpt])
#                     tar = np.array([targets[idx:idx + self.length] for idx in excerpt])
#
#                     yield inp, tar
#
#         else:  # IF dataset can fit the memory
#
#             # LOAD data from csv
#             data_frame = pd.read_csv(self.filename,
#                                      nrows=self.crop,
#                                      header=self.header
#                                      )
#
#             np_array = np.array(data_frame)
#             inputs, targets = np_array[:, 0], np_array[:, 1]
#
#             max_batchsize = inputs.size - self.length + 1
#             if self.batchsize < 0:
#                 self.batchsize = max_batchsize
#
#             # define indices and shuffle them if necessary
#             indices = np.arange(max_batchsize) #start默认为0，stop为max_batchsize，步长默认为1
#             if self.shuffle:
#                 np.random.shuffle(indices)
#
#             # providing sliding windows:
#             for start_idx in range(0, max_batchsize, self.batchsize): #[0,1000,2000,3000,.....max_batchsize]
#                 excerpt = indices[start_idx:start_idx + self.batchsize]
#
#                 inp = np.array([inputs[idx:idx + self.length] for idx in excerpt])  #这会得到一个ndarray
#                 #inp=[array([0, 1, 2,....599]) array([1, 2, ....600]) ......array([999, 1000,.....1598])]
#                 tar = np.array([targets[idx:idx + self.length] for idx in excerpt])
#
#                 yield inp, tar