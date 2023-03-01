from Arguments import *
from Logger import log
from tensorflow.keras.models import Model  # Input,
from tensorflow.keras.layers import Dense, Dropout,SeparableConv2D,GRU,Bidirectional, Conv2D, GlobalMaxPooling2D, Flatten, Reshape, Lambda, dot, \
    UpSampling2D, Add, Concatenate, Activation, concatenate, Conv1D, SpatialDropout1D, BatchNormalization, add,MaxPooling2D
from tensorflow.keras.utils import plot_model  # print_summary,
import numpy as np
import tensorflow.keras.backend as K
import os
from typing import List, Tuple
import tensorflow.keras.backend as K
# #import keras.layers
# from tensorflow.keras import optimizers
# from tensorflow.keras.engine.topology import Layer
# import tensorflow as tf # if tensorflow 1
# import tensorflow.compat.v1 as tf # if using tensorflow 2
# tf.disable_v2_behavior()
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

########################
import h5py
import argparse
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, \
    Concatenate,Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.activations import sigmoid

# import tensorflow  as tf

# Model setting begin, used in Sequence to point Learning based on bidirectional GRU and PAN network for nilm  
# --------------------------------
# -------------seq2point baseline
def S2P_model(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    cnn1 = Conv2D(filters=30,
                  kernel_size=(10, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(reshape)

    cnn2 = Conv2D(filters=30,
                  kernel_size=(8, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn1)

    cnn3 = Conv2D(filters=40,
                  kernel_size=(6, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn2)

    cnn4 = Conv2D(filters=50,
                  kernel_size=(5, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn3)

    cnn5 = Conv2D(filters=50,
                  kernel_size=(5, 1),
                  strides=(1, 1),
                  padding='same',
                  activation='relu',
                  )(cnn4)

    flat = Flatten(name='flatten')(cnn5)

    d = Dense(1024, activation='relu', name='dense')(flat)

    # if n_dense == 1:
    #     label = d
    # elif n_dense == 2:
    #     d1 = Dense(1024, activation='relu', name='dense1')(d)
    #     label = d1
    # elif n_dense == 3:
    #     d1 = Dense(1024, activation='relu', name='dense1')(d)
    #     d2 = Dense(1024, activation='relu', name='dense2')(d1)
    #     label = d2

    d_out = Dense(1, activation='linear', name='output')(d)

    model = Model(inputs=input_tensor, outputs=d_out)
    # Model setting done
    ####model structure done!
    ##############################
    # session = K.get_session() # For Tensorflow 1
    session = tf.keras.backend.get_session()  # For Tensorflow 2
    #   The name tf.keras.backend.get_session is deprecated. Please use tf.compat.v1.keras.backend.get_session instead.
    ##############################
    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')
    # Printing, logging and plotting the model
    # print_summary(model_def)
    model_def.summary()
    # plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def

def C2(x, channel):
    layer2 = Conv2D(filters=channel * 2,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(x)
    return layer2


def C21(x, channel, n):
    layer1 = C2(x, channel)
    return layer1


def C_lateral(x, channel):
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    # padding='same',
                    activation='relu',
                    )(x)
    return layer1


def C_up(x, channel):
    layer1 = UpSampling2D((1, 2))(x)
    layer2 = Conv2D(filters=channel,
                    kernel_size=(1, 2),
                    strides=(1, 1),
                    activation='relu',
                    )(layer1)
    return layer2

# -----------------------------------------------
def AugLPN_NILM(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
               cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    channel = 32
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(reshape)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = Add()([UP1, layer3_lateral])
    layer2_lateral = C_lateral(layer2, channel)
    UP2 = UpSampling2D((1, 2))(layer3_)
    layer2_ = Add()([UP2, layer2_lateral])
    layer1_lateral = C_lateral(layer1, channel)
    UP3 = UpSampling2D((1, 2))(layer2_)
    layer1_ = Add()([UP3, layer1_lateral])
    layer1_temp_condv = Conv2D(filters=channel,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               )(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = GlobalMaxPooling2D()(layer1_)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Conv2D(filters=channel,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='relu'
                      )(max_pool)

    max_pool = Activation('hard_sigmoid')(max_pool)
    layer1_mul = multiply([max_pool, layer1_temp_condv])

    layer1_b3 = Conv2D(filters=channel,
                       kernel_size=(1, 3),
                       strides=(1, 1),
                       padding='same'
                       )(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)

    layer1_mul = multiply([layer1_mul, result_1])

    layer2_up = Conv2D(filters=channel * 2,
                       kernel_size=(1, 2),
                       strides=(1, 2)
                       )(layer1_mul)
    layer2_pan_ = Conv2D(filters=channel * 2,
                         kernel_size=(1, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(layer2_)
    layer3_pan = Add()([layer2_up, layer2_pan_])

    layer3_pan_ = Conv2D(filters=channel,
                         kernel_size=(1, 3),
                         strides=(1, 2),
                         padding='same',
                         activation='relu',
                         )(layer3_pan)
    pan_Preoutput = Add()([layer3_pan_, layer3_])

    layer3_pan = Conv2D(filters=96, 
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=2
                        )(layer3_pan)
    layer3_pan = Conv2D(filters=64,  
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=4
                        )(layer3_pan)
    layer3_pan = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan) 
    layer3_pan = Conv2D(filters=32,
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu'
                        )(layer3_pan)

    pan_Preoutput = Add()([layer3_pan, pan_Preoutput]) 
    
    layer3_total_SepConv1 = SeparableConv2D(filters=128, kernel_size=(1, 2),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)
    layer3_total_SepConv2 = SeparableConv2D(filters=128, kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)

    Sep_Add = Add()([layer3_total_SepConv1, layer3_total_SepConv2]) 

    Dil_Add = Reshape((75, 128), )(Sep_Add)
    BiGru2 = Bidirectional(GRU(64, return_sequences=True))(Dil_Add)
    Gru2_output = Reshape((-1, 75, 128), )(BiGru2)
    Gru2_output = Conv2D(filters=64,
                            kernel_size=(1, 2),
                            strides=(1, 1),
                            padding='same',
                            activation='relu',
                            )(Gru2_output)
    layer3_total_SepConv1_1 = SeparableConv2D(filters=64, kernel_size=(1, 2),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    layer3_total_SepConv2_2 = SeparableConv2D(filters=64, kernel_size=(1, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    Sep_Add_1 = Add()([layer3_total_SepConv1_1, layer3_total_SepConv2_2]) 
    Sep_Add_1 = Conv2D(filters=64,
                            kernel_size=(1, 2),
                            strides=(1, 2),
                            padding='same',
                            activation='relu',
                            )(Sep_Add_1)
    Dil_Add_1 = Reshape((75, 64), )(Sep_Add_1)
    BiGru2_2 = Bidirectional(GRU(32, return_sequences=True))(Dil_Add_1)
    Gru2_output_1 = Reshape((-1, 75, 64), )(BiGru2_2)
    right_output = Add()([Gru2_output, Gru2_output_1])
    out1 = Concatenate()([right_output, pan_Preoutput]) 
    flat = Flatten(name='flatten')(out1)
    dense1 = Dense(56, activation='relu', name='dense1')(flat)
    dense1 = Dense(10, activation='relu', name='dense2')(dense1)
    d_out = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = tf.keras.backend.get_session()  # For Tensorflow 2

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    model_def.summary()

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def
# ------------------------------------------------
# -----------------------------------------------
def AugLPN_NILM_16(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    channel = 16
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(reshape)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = Add()([UP1, layer3_lateral])
    layer2_lateral = C_lateral(layer2, channel)
    UP2 = UpSampling2D((1, 2))(layer3_)
    layer2_ = Add()([UP2, layer2_lateral])
    layer1_lateral = C_lateral(layer1, channel)
    UP3 = UpSampling2D((1, 2))(layer2_)
    layer1_ = Add()([UP3, layer1_lateral])

    # ----------PAN----------------------------------------------
    layer1_temp_condv = Conv2D(filters=channel,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               )(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = GlobalMaxPooling2D()(layer1_)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Conv2D(filters=channel,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='relu'
                      )(max_pool)

    max_pool = Activation('hard_sigmoid')(max_pool)
    layer1_mul = multiply([max_pool, layer1_temp_condv])

    layer1_b3 = Conv2D(filters=channel,
                       kernel_size=(1, 3),
                       strides=(1, 1),
                       padding='same'
                       )(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)

    layer1_mul = multiply([layer1_mul, result_1])

    layer2_up = Conv2D(filters=channel * 2,
                       kernel_size=(1, 2),
                       strides=(1, 2)
                       )(layer1_mul)

    layer2_pan_ = Conv2D(filters=channel * 2,
                         kernel_size=(1, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(layer2_)
    layer3_pan = Add()([layer2_up, layer2_pan_])

    layer3_pan_ = Conv2D(filters=channel,
                         kernel_size=(1, 3),
                         strides=(1, 2),
                         padding='same',
                         activation='relu',
                         )(layer3_pan)
    pan_Preoutput = Add()([layer3_pan_, layer3_])

    layer3_pan = Conv2D(filters=48,  
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=2
                        )(layer3_pan)

    layer3_pan = Conv2D(filters=32, 
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=4
                        )(layer3_pan)

    layer3_pan = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan)  
    layer3_pan = Conv2D(filters=channel,
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu'
                        )(layer3_pan)

    pan_Preoutput = Add()([layer3_pan, pan_Preoutput]) 

    layer3_total_SepConv1 = SeparableConv2D(filters=64, kernel_size=(1, 2),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)
    layer3_total_SepConv2 = SeparableConv2D(filters=64, kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)

    Sep_Add = Add()([layer3_total_SepConv1, layer3_total_SepConv2]) 
    Dil_Add = Reshape((75, 64), )(Sep_Add)
    BiGru2 = Bidirectional(GRU(32, return_sequences=True))(Dil_Add)

    Gru2_output = Reshape((-1, 75, 64), )(BiGru2)
    Gru2_output = Conv2D(filters=32,
                         kernel_size=(1, 2),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(Gru2_output)

    layer3_total_SepConv1_1 = SeparableConv2D(filters=32, kernel_size=(1, 2),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    layer3_total_SepConv2_2 = SeparableConv2D(filters=32, kernel_size=(1, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)

    Sep_Add_1 = Add()([layer3_total_SepConv1_1, layer3_total_SepConv2_2]) 
    Sep_Add_1 = Conv2D(filters=32,
                       kernel_size=(1, 2),
                       strides=(1, 2),
                       padding='same',
                       activation='relu',
                       )(Sep_Add_1)

    Dil_Add_1 = Reshape((75, 32), )(Sep_Add_1)
    BiGru2_2 = Bidirectional(GRU(16, return_sequences=True))(Dil_Add_1)

    Gru2_output_1 = Reshape((-1, 75, 32), )(BiGru2_2)


    right_output = Add()([Gru2_output, Gru2_output_1])

    out1 = Concatenate()([right_output, pan_Preoutput])  

    flat = Flatten(name='flatten')(out1)
    dense1 = Dense(28, activation='relu', name='dense1')(flat)
    # dense1 = Dropout(0.1)(dense1)
    dense1 = Dense(10, activation='relu', name='dense2')(dense1)
    d_out = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = tf.keras.backend.get_session()  # For Tensorflow 2

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    model_def.summary()

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def
# ------------------------------------------------

# -----------------------------------------------
def AugLPN_NILM_48(appliance, input_tensor, window_length, transfer_dense=False, transfer_cnn=False,
              cnn='fridge', pretrainedmodel_dir='./models/', n_dense=1):
    reshape = Reshape((-1, window_length, 1), )(input_tensor)
    channel = 48
    layer1 = Conv2D(filters=channel,
                    kernel_size=(1, 3),
                    strides=(1, 2),
                    padding='same',
                    activation='relu',
                    )(reshape)

    layer2 = C21(layer1, channel, 2)
    layer3 = C21(layer2, channel * 2, 2)
    layer4 = C21(layer3, channel * 4, 2)

    layer4_lateral = C_lateral(layer4, channel)
    UP1 = C_up(layer4_lateral, channel)
    layer3_lateral = C_lateral(layer3, channel)
    layer3_ = Add()([UP1, layer3_lateral])
    layer2_lateral = C_lateral(layer2, channel)
    UP2 = UpSampling2D((1, 2))(layer3_)
    layer2_ = Add()([UP2, layer2_lateral])
    layer1_lateral = C_lateral(layer1, channel)
    UP3 = UpSampling2D((1, 2))(layer2_)
    layer1_ = Add()([UP3, layer1_lateral])

    # ----------PAN----------------------------------------------
    layer1_temp_condv = Conv2D(filters=channel,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               )(layer1_)
    layer1_temp_condv = tf.nn.l2_normalize(layer1_temp_condv, axis=2)

    max_pool = GlobalMaxPooling2D()(layer1_)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = Conv2D(filters=channel,
                      kernel_size=(1, 1),
                      strides=(1, 1),
                      activation='relu'
                      )(max_pool)

    max_pool = Activation('hard_sigmoid')(max_pool)
    layer1_mul = multiply([max_pool, layer1_temp_condv])

    layer1_b3 = Conv2D(filters=channel,
                       kernel_size=(1, 3),
                       strides=(1, 1),
                       padding='same'
                       )(layer1_)
    result_1 = tf.nn.l2_normalize(layer1_b3, axis=2)

    layer1_mul = multiply([layer1_mul, result_1])

    layer2_up = Conv2D(filters=channel * 2,
                       kernel_size=(1, 2),
                       strides=(1, 2)
                       )(layer1_mul)

    layer2_pan_ = Conv2D(filters=channel * 2,
                         kernel_size=(1, 3),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(layer2_)
    layer3_pan = Add()([layer2_up, layer2_pan_])

    layer3_pan_ = Conv2D(filters=channel,
                         kernel_size=(1, 3),
                         strides=(1, 2),
                         padding='same',
                         activation='relu',
                         )(layer3_pan)
    pan_Preoutput = Add()([layer3_pan_, layer3_])

    layer3_pan = Conv2D(filters=144, 
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=2
                        )(layer3_pan)
    layer3_pan = Conv2D(filters=96,  
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu',
                        dilation_rate=4
                        )(layer3_pan)

    layer3_pan = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(layer3_pan)
    layer3_pan = Conv2D(filters=channel,
                        kernel_size=(1, 3),
                        strides=(1, 1),
                        padding='same',
                        activation='relu'
                        )(layer3_pan)

    pan_Preoutput = Add()([layer3_pan, pan_Preoutput]) 

    layer3_total_SepConv1 = SeparableConv2D(filters=192, kernel_size=(1, 2),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)
    layer3_total_SepConv2 = SeparableConv2D(filters=192, kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            activation='relu')(layer3)

    Sep_Add = Add()([layer3_total_SepConv1, layer3_total_SepConv2]) 

    Dil_Add = Reshape((75, 192), )(Sep_Add)
    BiGru2 = Bidirectional(GRU(96, return_sequences=True))(Dil_Add)

    Gru2_output = Reshape((-1, 75, 192), )(BiGru2)
    Gru2_output = Conv2D(filters=96,
                         kernel_size=(1, 2),
                         strides=(1, 1),
                         padding='same',
                         activation='relu',
                         )(Gru2_output)

    layer3_total_SepConv1_1 = SeparableConv2D(filters=96, kernel_size=(1, 2),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)
    layer3_total_SepConv2_2 = SeparableConv2D(filters=96, kernel_size=(1, 3),
                                              strides=(1, 1),
                                              padding='same',
                                              activation='relu')(layer2)

    Sep_Add_1 = Add()([layer3_total_SepConv1_1, layer3_total_SepConv2_2]) 
    Sep_Add_1 = Conv2D(filters=96,
                       kernel_size=(1, 2),
                       strides=(1, 2),
                       padding='same',
                       activation='relu',
                       )(Sep_Add_1)

    Dil_Add_1 = Reshape((75, 96), )(Sep_Add_1)
    BiGru2_2 = Bidirectional(GRU(48, return_sequences=True))(Dil_Add_1)

    Gru2_output_1 = Reshape((-1, 75, 96), )(BiGru2_2)

    right_output = Add()([Gru2_output, Gru2_output_1])

    out1 = Concatenate()([right_output, pan_Preoutput])

    flat = Flatten(name='flatten')(out1)
    dense1 = Dense(84, activation='relu', name='dense1')(flat)
    # dense1 = Dropout(0.1)(dense1)
    dense1 = Dense(10, activation='relu', name='dense2')(dense1)
    d_out = Dense(1, activation='linear', name='output')(dense1)

    model = Model(inputs=input_tensor, outputs=d_out)

    session = tf.keras.backend.get_session()  # For Tensorflow 2

    # For transfer learning
    if transfer_dense:
        log("Transfer learning...")
        log("...loading an entire pre-trained model")
        weights_loader(model, pretrainedmodel_dir + '/cnn_s2p_' + appliance + '_pointnet_model')
        model_def = model
    elif transfer_cnn and not transfer_dense:
        log("Transfer learning...")
        log('...loading a ' + appliance + ' pre-trained-cnn')
        cnn_weights_loader(model, cnn, pretrainedmodel_dir)
        model_def = model
        for idx, layer1 in enumerate(model_def.layers):
            if hasattr(layer1, 'kernel_initializer') and 'conv2d' not in layer1.name and 'cnn' not in layer1.name:
                log('Re-initialize: {}'.format(layer1.name))
                layer1.kernel.initializer.run(session=session)

    elif not transfer_dense and not transfer_cnn:
        log("Standard training...")
        log("...creating a new model.")
        model_def = model
    else:
        raise argparse.ArgumentTypeError('Model selection error.')

    model_def.summary()

    # Adding network structure to both the log file and output terminal
    files = [x for x in os.listdir('./') if x.endswith(".log")]
    with open(max(files, key=os.path.getctime), 'a') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model_def.summary(print_fn=lambda x: fh.write(x + '\n'))
    return model_def
# ------------------------------------------------

def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))


def cnn_weights_loader(model_to_fill, cnn_appliance, pretrainedmodel_dir):
    log('Loading cnn weights from ' + cnn_appliance)
    weights_path = pretrainedmodel_dir + '/cnn_s2p_' + cnn_appliance + '_pointnet_model' + '_weights.h5'
    if not os.path.exists(weights_path):
        print('The directory does not exist or you do not have the files for trained model')

    f = h5py.File(weights_path, 'r')
    log(f.visititems(print_attrs))
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    for name in layer_names:
        if 'conv2d_' in name or 'cnn' in name:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]

            model_to_fill.layers[int(name[-1]) + 1].set_weights(weight_values)
            log('Loaded cnn layer: {}'.format(name))

    f.close()
    print('Model loaded.')


def weights_loader(model, path):
    log('Loading cnn weights from ' + path)
    model.load_weights(path + '_weights.h5')
