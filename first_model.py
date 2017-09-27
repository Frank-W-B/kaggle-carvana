import numpy as np
import pandas as pd
import sys
import os
from skimage.io import imread
from matplotlib import pyplot as plt
from keras import backend as K
import time
from architecture_flexible import set_architecture

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

def stich_two_imgs_together(img1, img2):
    assert (img1.shape[0] == img2.shape[0]), 'Images are not the same height'
    lw_bk = 2
    lw_wt = 20
    vline_bk = np.zeros((img1.shape[0], lw_bk, img1.shape[2]), dtype = 'uint8')
    vline_wt = np.ones((img1.shape[0], lw_wt, img1.shape[2]), dtype = 'uint8') * 255
    img = np.hstack((img1, vline_bk))
    img = np.hstack((img, vline_wt))
    img = np.hstack((img, vline_bk))
    img = np.hstack((img, img2))
    return img

def make_1_channel_into_3_channel_img(img_1_chan):
    img_3_channel = np.zeros((img_1_chan.shape[0], img_1_chan.shape[1], 3), dtype = 'uint8')
    img_3_channel[:,:,0] = img_1_chan
    img_3_channel[:,:,1] = img_1_chan
    img_3_channel[:,:,2] = img_1_chan
    return img_3_channel

def dice_coefficient(img_true, img_pred):
    img_pred = img_pred > 0.5
    img_pred = img_pred.astype(int)
    img_true = img_true > 0.5
    img_true = img_true.astype(int)
    TP = sum(sum(np.multiply(img_true, img_pred)))
    P = sum(sum(img_true))
    PP = sum(sum(img_pred))
    FP = PP - TP
    dice = round((TP / (P + FP)), 3)
    return dice

if __name__ == '__main__':
    img_w = 256       # img width (pixels), 256 original
    img_h = 256       # img height (pixels)
    n_labels = 2      # number of labels
    img_channels = 3
    batch_size = 1 
    n_epochs = 500

    X_train = np.load('X_train.npy') 
    y_train = np.load('y_train.npy')
   
    X_test = np.load('X_test.npy') 
    y_test = np.load('y_test.npy')

    input_shape = (img_channels, img_h, img_w) # channels first
    layers_in_block = [1] 
    for l in layers_in_block:
        print("Layers in block: {0}".format(l))
        model = set_architecture(n_labels, input_shape, conv_layers_in_block=l)
        #optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        print('Compiled')
        t0 = time.time()
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epochs,
                        validation_data = (X_test, y_test), verbose=1) 
        t1 = time.time()
        elapsed = int(round(t1-t0, 0))
        #model.save_weights('weights.hdf5')
        #model.load_weights('model_5l_weight_ep50.hdf5')

        score = model.evaluate(X_test, y_test, verbose=0)
        loss = round(score[0], 3)
        accuracy = round(score[1], 3)
        print("Test loss: {0}".format(loss))
        print("Test accuracy: {0}".format(accuracy))
        print("Computation time: {0}".format(elapsed))
    
        # for images
        y_pred = model.predict(X_test, verbose=0)
        y_pred = y_pred.reshape((y_pred.shape[0], img_h, img_w, n_labels))
        y_true = y_test.reshape((y_test.shape[0], img_h, img_w, n_labels))
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_train = y_pred_train.reshape((y_pred_train.shape[0], img_h, img_w, n_labels))
        y_true_train = y_train.reshape((y_train.shape[0], img_h, img_w, n_labels))

        pic_y_true = y_true[0][:,:,1] * 255
        pic_y_pred = y_pred[0][:,:,1] * 255
        pic_y_true_train = y_true_train[0][:,:,1] * 255
        pic_y_pred_train = y_pred_train[0][:,:,1] * 255
        pic_y_true = make_1_channel_into_3_channel_img(pic_y_true)
        pic_y_pred = make_1_channel_into_3_channel_img(pic_y_pred)
        pic_y_true_train = make_1_channel_into_3_channel_img(pic_y_true_train)
        pic_y_pred_train = make_1_channel_into_3_channel_img(pic_y_pred_train)
        
        pic_X_train = X_train[0]
        pic_X_train = pic_X_train.reshape((img_h, img_w, img_channels))
        pic_X_test = X_test[0]
        pic_X_test = pic_X_test.reshape((img_h, img_w, img_channels))


        training = stich_two_imgs_together(pic_X_train, pic_y_true_train)
        training = stich_two_imgs_together(training, pic_y_pred_train)
        dice_train = dice_coefficient(pic_y_true_train[:,:,0], pic_y_pred_train[:,:,0])
        plt.imshow(training)
        plt.xticks([])
        plt.yticks([])
        fname_train = "Train_L-" + str(l) + "__e-" + str(n_epochs) + "__ac-" + str(accuracy) + \
                "--dice-" + str(dice_train) + "_.png"
        plt.savefig(fname_train)
        plt.close()

        testing = stich_two_imgs_together(pic_X_test, pic_y_true)
        testing = stich_two_imgs_together(testing, pic_y_pred)
        dice_test = dice_coefficient(pic_y_true[:,:,0], pic_y_pred[:,:,0])
        plt.imshow(testing)
        plt.xticks([])
        plt.yticks([])
        fname_test = "Test_L-" + str(l) + "__e-" + str(n_epochs) + "__ac-" + str(accuracy) + \
                "--dice-" + str(dice_test) + "_.png"
        plt.savefig(fname_test)
        plt.close()
