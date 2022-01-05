import os
import pandas as pd
import numpy as np
import skimage
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import pickle
from keras.models import load_model
import keras.backend as K
from keras import optimizers, applications, layers
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping
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


dependencies = {
    'auc_roc': auc_roc,
    'binary_precision': km.binary_precision(label=1),
    'binary_recall': km.binary_recall(label=1)
}
print('Reading csv and adding true labels...')
os.chdir('/home/steve/PycharmProjects/hyperfine/size-intensity/')
df = pd.read_csv('size_intensity_original.csv')
dataset_names = set(df['Dataset'])

df['True_Label']=~df['Image'].str.contains('nonlesion')
df['True_Label']=df['True_Label'].astype('int').astype('str')
df['Pred_Score']=np.nan
df['Pred_Label']=np.nan
print('done.')
# at the end, we can make a column for "correct" by just seeing if they match the label
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
for dataset in dataset_names:
    dataset = dataset.strip()
    print('Analyzing dataset: ' + dataset)

    if dataset=='ISLES':
        model = load_model('stroke_3t_model.h5', custom_objects=dependencies)
        dir = '/home/steve/PycharmProjects/hyperfine/stroke/ISLES_2015/'
        analysis_df = df[df['Dataset'].str.contains("ISLES")]
    elif dataset=='BRATS_LGG':
        model = load_model('lgg_3t_model.h5', custom_objects=dependencies)
        dir = '/home/steve/PycharmProjects/hyperfine/optimized_brats/BraTS_2019/LGG/'
        analysis_df = df[df['Dataset'].str.contains("BRATS_LGG")]
    elif dataset=='BRATS_HGG':
        model = load_model('hgg_3t_model.h5', custom_objects=dependencies)
        dir = '/home/steve/PycharmProjects/hyperfine/optimized_brats/BraTS_2019/HGG/'
        analysis_df = df[df['Dataset'].str.contains("BRATS_HGG")]
    elif dataset=='MS2008':
        model = load_model('ms_3t_model.h5', custom_objects=dependencies)
        dir='/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/MS_SEG_2008/'
        analysis_df = df[df['Dataset'].str.contains("MS2008")]
    elif dataset=='MS2016':
        model = load_model('ms_3t_model.h5', custom_objects=dependencies)
        dir='/home/steve/PycharmProjects/hyperfine/optimized_MS_10-21-19/MS_SEG_2016/'
        analysis_df = df[df['Dataset'].str.contains("MS2016")]
    else:
        print('DATASET NOT FOUND!')

    print('...dataset loaded')

    all_subs = sorted(os.listdir(dir))
    num_val_subs = int(np.floor(len(all_subs)/10))
    val_subs = all_subs[-1*num_val_subs:]

    #Here we fix some things that are only relevant for the 3t sheet
    analysis_df = analysis_df.replace(to_replace=r'\\', value='/', regex=True)

    if dataset=='ISLES':
        analysis_df['Image'] = 'sub' + analysis_df['Image'].astype(str)

    analysis_df = analysis_df[analysis_df['Image'].str.contains('|'.join(val_subs))]

    print('...validation patients selected')

    test_gen = val_datagen.flow_from_dataframe(dataframe=analysis_df, directory=dir,
                                              x_col='Image', y_col='True_Label', target_size=(224, 224), color_mode='rgb',
                                              class_mode='binary', batch_size=1, shuffle=False)
    preds = model.predict_generator(test_gen, steps=test_gen.n, verbose=1)

    analysis_df['Pred_Score'] = preds[:,1]
    analysis_df['Pred_Label'] = preds[:,1]>0.5

    #df = pd.merge(df, analysis_df, on=['Dataset','Image','Size','Intensity','True_Label'], how='outer')
    df.update(analysis_df)

    print('...predictions generated and added to dataframe')

#df['Pred_Label'] = df['Pred_Label'].astype('int').astype('str')
df['Accurate'] = df['True_Label'].astype('int').astype('bool')==df['Pred_Label']

print('ALL FINISHED.')
df.to_csv('validation_results_3t.csv', sep='\t', index=False)