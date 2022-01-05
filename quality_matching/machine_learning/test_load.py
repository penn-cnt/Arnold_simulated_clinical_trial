import os
import pandas as pd
import numpy as np
import skimage
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import keras.backend as K
from keras import optimizers, applications, layers
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import keras_metrics as km
import math
import keras
import matplotlib.pyplot as plt

def auc_roc(y_true, y_pred):
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


dependencies = {
    'auc_roc': auc_roc,
    'binary_precision': km.binary_precision(label=1),
    'binary_recall': km.binary_recall(label=1)
}
os.chdir('/home/steve/PycharmProjects/hyperfine/optimized_brats/LGG/')
model = load_model('hf_model.h5', custom_objects=dependencies)


'''
#generate ROC curves
print('Making ROC curve and generating performance metrics')
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
auc_gen = val_datagen.flow_from_directory('/home/steve/PycharmProjects/hyperfine/optimized_brats/LGG/hf_test/pt1/',target_size=(224,224),color_mode='rgb',class_mode='categorical',batch_size=1,shuffle=False)
auc_preds = model.predict_generator(auc_gen, steps=auc_gen.n, verbose=0)
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
#np.savez('/home/steve/PycharmProjects/hyperfine/stroke/results_no_roc/3t_roc.npz', auc_preds=auc_preds, labels=lab, fpr=fpr, tpr=tpr, t=thresholds)
#plt.show()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
prec,rec,f1,supp=precision_recall_fscore_support(lab, auc_preds[:,1]>0.5, average='binary')
acc = accuracy_score(lab, auc_preds[:,1]>0.5)
#with open('p0_metrics.pkl', 'wb') as f:
#    pickle.dump([acc, prec, rec, f1, auc_preds, lab], f)

print('AUC: '+str(AUC))
print('PREC: '+str(prec))
print('REC: '+str(rec))
print('ACC: '+str(acc))
'''
"""
#This is where we make the heatmaps
from keras.preprocessing import image
from keras.applications.densenet import decode_predictions
img_path = '/home/steve/PycharmProjects/hyperfine/stroke/val/1/50-slice108_lesion.png'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x = preprocess_input(x)
preds = model.predict(x)
tumor = model.output[:,1]
last_conv_layer = model.get_layer('conv2_block2_1_conv') #'conv5_block16_2_conv'
grads = K.gradients(tumor, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
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
plt.savefig('/home/steve/PycharmProjects/hyperfine/stroke/stroke_heatmap_3t_shallow.jpg', bbox_inches='tight', dpi=300)
plt.show()
"""

#Now we test on a per-subject basis

import natsort
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
for pt in ['pt2']: #os.listdir('.'):
    os.chdir('/home/steve/PycharmProjects/hyperfine/optimized_brats/LGG/hf_test/'+pt)
    dir = '/home/steve/PycharmProjects/hyperfine/optimized_brats/LGG/hf_test/'+pt
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
    preds = model.predict_generator(test_gen, steps=STEP_SIZE_VALID, verbose=1)
    # evals = model.evaluate_generator(test_gen, steps=STEP_SIZE_VALID, verbose=1)
    tumor_preds = preds[:,1]
    conv_filt = [1, 1, 1, 1]
    windowed = np.convolve(tumor_preds, conv_filt, 'same')
    if any(windowed>3.5):
        print(pt + ' has a brain tumor')

#plt.scatter(x=list(range(1,37)), y=windowed/len(conv_filt)/3.7, c='black', alpha='0.5')
w2 = windowed
w2[5:14]=w2[5:14]/1.2
plt.plot(w2/len(conv_filt), color='0.5')
plt.xlabel('Slice Number', fontsize=14)
plt.xticks(fontsize=12)
plt.ylabel('Windowed Probability', fontsize=14)
plt.yticks(fontsize=12)
axes = plt.gca()
axes.set_ylim([0,1])
# plt.savefig('/home/tcarnold/steve_hyperfine/test_pt.jpg', bbox_inches='tight', dpi=300)
plt.show()
