import os
import pandas as pd
import numpy as np
import skimage
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pickle
from keras.models import load_model
import keras.backend as K
from keras import optimizers, applications, layers
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import keras_metrics as km
import math
import keras

def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

#def new_auc(y_true, y_pred) :
#    if len(np.unique(y_true)) == 1:
#        return 0
#    score = tf.py_func( lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),
#                        [y_true, y_pred],
#                        'float32',
#                        stateful=False,
#                        name='sklearnAUC' )
#    return score

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def mean_label(y_true, y_pred):
    return K.mean(y_true)

class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result

#K.set_learning_phase(1)
import logging
LOG = "hyperfine.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.ERROR)
# console handler
console = logging.StreamHandler()
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

K.clear_session()

print('starting')

#include this for the intensity modulation part, and add to paths:
#per = 'p100'


os.chdir('/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/')
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

class_list = ["0.0", "1.0"]
#directory='/home/steve/PycharmProjects/mimic-cxr/data',
#directory='/media/steve/Samsung_T5/MIMICCXR',
train_gen = train_datagen.flow_from_directory('/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/train/',target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=32,shuffle=True)
val_gen = val_datagen.flow_from_directory('/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/val/',target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=32,shuffle=True)

print('initialized data generator')

print('making model')
# Networks
from keras.applications.densenet import DenseNet121

# Layers
from keras.layers import *


#class FrozenBatchNormalization(layers.BatchNormalization):
#    def call(self, inputs, training=None):
#        return super().call(inputs=inputs, training=False)


#BatchNormalization = layers.BatchNormalization
#layers.BatchNormalization = FrozenBatchNormalization

#Prepare the Model
HEIGHT = 224
WIDTH = 224
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3), classes=2)

#undo the patch
#layers.BatchNormalization = BatchNormalization

"""
for layer in base_model.layers[:115]:
    layer.trainable = False
for layer in base_model.layers[115:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
dropout=0.5
x = Dense(200, activation='relu')(x)
x = Dropout(dropout)(x)
prior_probability=0.05
x = Dense(50, activation='relu',bias_initializer=initializers.PriorProbability(probability=prior_probability))(x)
x = Dropout(dropout)(x)
num_classes = len(class_list)
predictions = Dense(num_classes, activation='softmax')(x)
finetune_model = Model(inputs=base_model.input, outputs=predictions)
"""
### build without prior probability set
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    #for layer in base_model.layers:
    #    layer.trainable = False
    #for layer in base_model.layers[:126]:
    #    layer.trainable=False
    #for layer in base_model.layers[126:]:
    #    layer.trainable=True

    x = base_model.output
    #x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

FC_LAYERS = [512, 512, 512, 512]
dropout = 0.5

finetune_model = build_finetune_model(base_model,
                                      dropout=dropout,
                                      fc_layers=FC_LAYERS,
                                      num_classes=len(class_list))
# end model building section

recall = km.binary_recall(label=1)
precision = km.binary_precision(label=1)

finetune_model.compile(loss='binary_crossentropy',
                       optimizer='nadam',
                       metrics=['accuracy', auc_roc, km.binary_precision(label=1), km.binary_recall(label=1)])

print('fitting model')
class_weights = {0: 1.,
                 1: 1.}

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=val_gen.n//val_gen.batch_size
es = EarlyStopping(monitor='val_loss',patience=50,verbose=1,restore_best_weights=True)

history= finetune_model.fit_generator(
        generator=train_gen,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=300,
        callbacks=[es],
        validation_data=val_gen,
        validation_steps=STEP_SIZE_VALID,
        workers=6,
        class_weight=class_weights,
        verbose=1)

print('evaluate phase I model and save to log file')
e = finetune_model.evaluate_generator(val_gen, steps=STEP_SIZE_VALID, verbose=0)
logger.error('Phase I: ' + str(e))
p = finetune_model.predict_generator(val_gen, steps=STEP_SIZE_VALID, verbose=0)
import matplotlib.pyplot as plt
#plt.plot(p[:,1])
#plt.show()
"""
#Now we test on a per-subject basis

os.chdir('/home/steve/PycharmProjects/hyperfine/optimized_brats/normal_HF_like')
import natsort
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
for pt in ['sub1']: #os.listdir('.'):
    os.chdir('/home/steve/PycharmProjects/hyperfine/optimized_brats/normal_HF_like/'+pt)
    dir = '/home/steve/PycharmProjects/hyperfine/optimized_brats/normal_HF_like/'+pt
    print('predicting on '+pt)
    imgs = natsort.natsorted(os.listdir('.'))
    # make dataframe with the slides in order
    df = pd.DataFrame(imgs)
    for i in imgs:
        if 'nonlesion' in i:
            df.loc[imgs.index(i), 1] = 0
        else:
            df.loc[imgs.index(i), 1] = 1
    df.loc[1,1]=0
    df.columns = ['img', 'label']
    df['label']=df['label'].astype(str)
    test_gen = test_datagen.flow_from_dataframe(df, directory=dir, x_col='img', y_col='label', target_size=(224, 224),
                                    color_mode='rgb', class_mode='categorical', batch_size=1, shuffle=False)
    STEP_SIZE_VALID = test_gen.n // test_gen.batch_size
    preds = finetune_model.predict_generator(test_gen, steps=STEP_SIZE_VALID, verbose=1)
    evals = finetune_model.evaluate_generator(test_gen, steps=STEP_SIZE_VALID, verbose=1)
    tumor_preds = preds[:,1]
    conv_filt = [1, 1, 1, 1]
    windowed = np.convolve(tumor_preds, conv_filt, 'same')
    if any(windowed>3.5):
        print(pt + ' has a brain tumor')

plt.plot(windowed/len(conv_filt)/3.7, color='0.5')
plt.xlabel('Slice Number', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Windowed Probability', fontsize=14)
plt.yticks(fontsize=12)
axes = plt.gca()
axes.set_ylim([0,1])
plt.savefig('/home/steve/PycharmProjects/hyperfine/optimized_brats/test_pt.jpg', bbox_inches='tight', dpi=300)
plt.show()
"""

"""
This is probably trash
import cv2
img = cv2.imread('/home/steve/PycharmProjects/hyperfine/data/hf_test/pt4/slice23_tumor.png')   # reads an image in the BGR format
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # BGR -> RGB
new_shape = (224, 224)
img2 = cv2.resize(img, new_shape)
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
img3 = preprocess_input(img2)
from keract import display_heatmaps, get_activations, display_activations
activations = get_activations(finetune_model, np.expand_dims(img3, axis=0), "conv_block1_1_conv")
display_activations(activations, cmap='gray', save=False)
display_heatmaps(activations,img2,save=False)

mat = np.squeeze(list(activations.values())[0])
m = np.max(mat,axis=2)
"""

"""
#This is where we make the heatmaps
from keras.preprocessing import image
from keras.applications.densenet import decode_predictions
img_path = '/home/steve/PycharmProjects/hyperfine/stroke/val/1/52-slice92_lesion.png'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x = preprocess_input(x)
preds = finetune_model.predict(x)
tumor = finetune_model.output[:,1]
last_conv_layer = finetune_model.get_layer('conv5_block1_1_conv') #'conv5_block16_2_conv'
grads = K.gradients(tumor, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([finetune_model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap *= -1
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
#plt.matshow(heatmap)
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255*heatmap)
heatmap = 255-heatmap
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#superimposed_img = heatmap* 0.4 + bw_img

from skimage import data, color, io, img_as_float

# Convert the input image and color mask to Hue Saturation Value (HSV)
# colorspace
img_hsv = color.rgb2hsv(img)
color_mask_hsv = color.rgb2hsv(heatmap)

# Replace the hue and saturation of the original image
# with that of the color mask
img_hsv[..., 0] = color_mask_hsv[..., 0]
img_hsv[..., 1] = color_mask_hsv[..., 1] * 0.6

img_masked = color.hsv2rgb(img_hsv)

#cv2.imwrite('/home/steve/PycharmProjects/hyperfine/heatmap.jpg',img_masked)
plt.imshow(img_masked)
plt.savefig('/home/steve/PycharmProjects/hyperfine/stroke/stroke_heatmap.jpg', bbox_inches='tight', dpi=300)
plt.show()
"""

#generate ROC curves
print('Making ROC curve and generating performance metrics')
auc_gen = val_datagen.flow_from_directory('/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/val/',target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=1,shuffle=False)
auc_preds = finetune_model.predict_generator(auc_gen, steps=auc_gen.n, verbose=0)
lab = auc_gen.classes
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(lab, auc_preds[:,1])
AUC = auc(fpr, tpr)
#plt.plot(fpr,tpr,color='0.5')
#plt.xlabel('False Positive Rate', fontsize=14)
#plt.ylabel('True Positive Rate', fontsize=14)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.savefig('/home/steve/PycharmProjects/hyperfine/brats_isointense/'+per+'/'+per+'_ROC.png',bbox_inches='tight', dpi=300)
np.savez('/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/clin_roc2.npz', auc_preds=auc_preds, labels=lab, fpr=fpr, tpr=tpr, t=thresholds)
#plt.show()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
prec,rec,f1,supp=precision_recall_fscore_support(lab, auc_preds[:,1]>0.5, average='binary')
acc = accuracy_score(lab, auc_preds[:,1]>0.5)
with open('clin_metrics2.pkl', 'wb') as f:
    pickle.dump([acc, prec, rec, f1, auc_preds, lab], f)

print('AUC: '+str(AUC))
print('PREC: '+str(prec))
print('REC: '+str(rec))
print('ACC: '+str(acc))

#save model
finetune_model.save('clin_model2.h5')
