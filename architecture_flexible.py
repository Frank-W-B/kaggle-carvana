

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Layer, Activation, Reshape, Permute


def set_architecture(n_classes, img_shape, conv_layers_in_block=1):
    model = Sequential()
    conv_kernel = (3, 3) # changing will affect padding and stride
    init = 'he_normal'
    
    # Encoding
    # Block 1
    model.add(Conv2D(64, conv_kernel, padding='same', input_shape=img_shape, kernel_initializer=init, data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    for l in range(conv_layers_in_block-1):
        model.add(Conv2D(64, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
        
    # Block 2
    for l in range(conv_layers_in_block):
        model.add(Conv2D(128, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))
        
    # Block 3
    for l in range(conv_layers_in_block):
        model.add(Conv2D(256, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first'))

    # Block 4
    for l in range(conv_layers_in_block):
        model.add(Conv2D(512, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))

    # Decoding
    # Block 4 
    model.add(ZeroPadding2D((1,1), data_format='channels_first' ))
    model.add(Conv2D(512, conv_kernel, padding='valid', kernel_initializer=init, data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    for l in range(conv_layers_in_block-1):
        model.add(Conv2D(512, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))

    # Block 3
    model.add(UpSampling2D((2,2), data_format='channels_first'))
    model.add(ZeroPadding2D((1,1), data_format='channels_first'))
    model.add(Conv2D(256, conv_kernel, padding='valid', kernel_initializer=init, data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    for l in range(conv_layers_in_block-1):
        model.add(Conv2D(256, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))


    # Block 2
    model.add(UpSampling2D((2,2), data_format='channels_first'))
    model.add(ZeroPadding2D((1,1), data_format='channels_first'))
    model.add(Conv2D(128, conv_kernel, padding='valid', kernel_initializer=init, data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    for l in range(conv_layers_in_block-1):
        model.add(Conv2D(128, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))


    # Block 1
    model.add(UpSampling2D((2,2), data_format='channels_first'))
    model.add(ZeroPadding2D((1,1), data_format='channels_first'))
    model.add(Conv2D(64, conv_kernel, padding='valid', data_format='channels_first'))
    model.add(BatchNormalization(axis=1))
    for l in range(conv_layers_in_block-1):
        model.add(Conv2D(64, conv_kernel, padding='same', kernel_initializer=init, data_format='channels_first'))
        model.add(BatchNormalization(axis=1))

    model.add(Conv2D(n_classes, conv_kernel, padding='same', data_format='channels_first' )) 
    output_shape = model.output_shape
    outputHeight = output_shape[2]
    outputWidth = output_shape[3]

    model.add(Reshape((-1, outputHeight*outputWidth)))
    model.add(Permute((2, 1)))
    model.add(Activation('softmax'))

    return model




if __name__ == '__main__':
    n_classes = 2
    img_channels = 3 
    img_h = 256
    img_w = 256
    conv_layers_in_block = 1   
    input_shape = (img_channels, img_h, img_w)  
    
    model = set_architecture(n_classes, input_shape, conv_layers_in_block)


