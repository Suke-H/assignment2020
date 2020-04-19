# -*- coding: utf-8 -*-
"""mic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16e8NyHwwm4Qg0p_zio9xFZRa3riIpfQY

# 2. 医用画像分類
"""

import numpy as np
from PIL import Image
import os
import sys
from glob import glob
import cv2
import re

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras import backend as K

"""## 画像の読み込み"""

# 画像のnumpy形式と対応する正解ラベルを返す関数
def readImg(paths):

  N = len(paths)

  # 画像読み込み準備
  imgs = [[] for i in range(N)]
  # 正解データ作成
  imgs_targets = []

  for k, path in enumerate(paths):
    
    #label = 画像が入ってるフォルダ名
    label = os.path.basename(os.path.dirname(path))

    if label == "0":
	    imgs_targets.append(0)
    else:
      imgs_targets.append(1)

    imgs[k] = np.array(Image.open(path))

    sys.stderr.write('{}枚目\r'.format(k))
    sys.stderr.flush()

  sys.stderr.write('\n')

  imgs = np.array(imgs, dtype = "float32")
  imgs_targets = np.array(imgs_targets, dtype = "int32")

  return imgs, imgs_targets

# データセットのあるパス
main_path = "/content/drive/My Drive/Colab Notebooks/Dataset/"
train_path = main_path + "train/"
val_path = main_path + "val/"
test_path = main_path + "test/"

# 全画像のパス読み込み
train_paths = np.array(sorted(glob(train_path + "**/*.png"),
                              key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1])))
val_paths = np.array(sorted(glob(val_path + "**/*.png"),
                            key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1])))
test_paths = np.array(sorted(glob(test_path + "**/*.png"),
                             key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1])))

print(len(train_paths), len(val_paths), len(test_paths))
print(val_paths)

"""## 初めて画像を読み込むときに使用．  
一度npy形式に保存したら以下のセルをコメントアウト推奨．
"""

# # 画像読み込み
# x_train, y_train = readImg(train_paths)
# x_val, y_val = readImg(val_paths)
# x_test, y_test = readImg(test_paths)

# x_train, y_train = readImgrgb(train_paths)
# x_val, y_val = readImgrgb(val_paths)
# x_test, y_test = readImgrgb(test_paths)
 
# print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape)

# # npy形式で保存
# np.save(main_path + "x_train", x_train)
# np.save(main_path + "y_train", y_train)
# np.save(main_path + "x_val", x_val)
# np.save(main_path + "y_val", y_val)
# np.save(main_path + "x_test", x_test)
# np.save(main_path + "y_test", y_test)

# np.save(main_path + "x_train_rgb", x_train)
# np.save(main_path + "x_val_rgb", x_val)
# np.save(main_path + "x_test_rgb", x_test)

"""## npy形式から読み込み"""

# npy形式から読み込み
x_train, y_train = np.load(main_path + "x_train.npy"), np.load(main_path + "y_train.npy")
x_val, y_val = np.load(main_path + "x_val.npy"), np.load(main_path + "y_val.npy")
x_test, y_test = np.load(main_path + "x_test.npy"), np.load(main_path + "y_test.npy")

print(x_train.shape, x_val.shape, x_test.shape)

"""## 画像の前処理"""

# 正規化
x_train /= 255.0
x_val /= 255.0
x_test /= 255.0

# 訓練データをシャッフル
perm = np.random.permutation(x_train.shape[0])
x_train, y_train = x_train[perm], y_train[perm]

"""## ディープラーニングによる画像分類

CNNモデルを用いて画像分類を行う．  
グレースケール画像なので，CNNに通す前に次元を増やす必要がある．
"""

# パラメータ
num_classes = 2
img_rows, img_cols, channel = 224, 224, 1

# 検証用の正解ラベルを保存しておく
true_labels = y_test[:]

# 正解ラベルをone-hotに変換
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# CNN用に次元を追加
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channel, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], channel, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], channel, img_rows, img_cols)
    input_shape = (channel, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, channel)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel)
    input_shape = (img_rows, img_cols, channel)

"""## モデル定義
モデルはhttps://qiita.com/wataoka/items5c6766d3e1c674d61425  を参照  
損失関数はbinary_crossentropy, 最適化アルゴリズムはAdadeltaを使用
"""

def model_net():

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

# 学習済みモデルがあったら読み込み
if os.path.exists(main_path + 'models/model_net.h5'):
    model = keras.models.load_model(main_path + 'models/model_net.h5', compile=False)
    print("モデル読み込み")

# なかったら新しく定義
else:
    model = model_net()
    print("モデル新規作成")

# モデル出力
print(model.summary())

# モデルの設定
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# パラメータ
batch_size, epochs = 128, 50

# # 学習
# stack = model.fit(x_train, y_train,
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   verbose=1,
#                   validation_data=(x_val, y_val))

# モデルの保存
model.save(main_path + 'models/model_net.h5', include_optimizer=False)

# 評価
score = model.evaluate(x_test, y_test, verbose=1)
print(list(zip(model.metrics_names, score)))

# 検証データから推定したラベルを出力
predict_labels = model.predict_classes(x_test)

# 推定ラベルと正解ラベルの組を入力し, 混合行列confusion_mおよび各組に対するラベルconfusion_lを返す
# confusion_m = [TP, TN, FP, FN]
# confusion_l = [0, 3, 1, ...](TP=0, TN=1, FP=2, FN=3)
def make_confusion_m(predict_labels, true_labels):
    confusion_m = np.array([0, 0, 0, 0])
    confusion_l = []

    for (true, pred) in zip(true_labels, predict_labels):
        # TP
        if true==True and pred==True:
            confusion_m[0] += 1
            confusion_l.append(0)
        
        # TN
        elif true==False and pred==False:
            confusion_m[1] += 1
            confusion_l.append(1)

        # FP
        elif true==False and pred==True:
            confusion_m[2] += 1
            confusion_l.append(2)

        # FN
        elif true==True and pred==False:
            confusion_m[3] += 1
            confusion_l.append(3)

    return confusion_m, np.array(confusion_l)

# 混合行列
confusion_m, confusion_l = make_confusion_m(predict_labels, true_labels)
print(confusion_m)

# 不正解の検証データのインデックスを取り出す
negative_indices = np.where(confusion_l >= 2)
print(negative_indices)

# 不正解の検証データ
negative_files = test_paths[negative_indices]
print(negative_files)