#!/usr/bin/env python
# coding: utf-8

# ## 1. numpyを使った行列の四則演算 

# In[1]:


import numpy as np

A = np.array([[1,2],[3,4]])
B = np.array([[5,6], [7,8]])
k = 2

# 行列の和
print("A + B = \n{}".format(A + B))
# 行列の差
print("A - B = \n{}".format(A - B))
# 行列の積
print("A * B = \n{}".format(np.dot(A, B)))
# スカラー積
print("k * A = \n{}".format(k * A))
# アダマール積
print("A @ B = \n{}".format(A * B))


# ## 2. 画像の表示，縮小拡大，回転，二値化 

# In[2]:


import cv2
from matplotlib import pyplot as plt
from IPython.display import Image, display

# 画面表示のメソッド
# jupyter_notebook用
# def show_img(img):
#     height, width = img.shape[:2]
#     img = cv2.imencode('.png', img)[1]
#     display(Image(img))
#     print("size: ({}, {})".format(width, height))
    
# Python用
def show_img(img):
    height, width = img.shape[:2]
    print("size: ({}, {})".format(width, height))

    # RGB画像
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)

    # グレースケール画像
    else:
        plt.imshow(img, cmap="gray")

    plt.show()
    plt.close()

# In[3]:


# 画像の読み込み
ori_img = cv2.imread("Csg_tree.png")
# サイズ取得
height, width = ori_img.shape[:2]
# 画像を2倍に拡大
big_img = cv2.resize(ori_img, (width*2,height*2))
# 画像を2分の1に縮小
small_img = cv2.resize(ori_img,(width//2, height//2))
# 画像を反時計回りに90度回転
rotate_img = cv2.rotate(ori_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

## 二値化
# グレースケール画像に変換
gray_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
# 閾値
threshold = 128
# 二値化
_, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)

# 画像表示
show_img(ori_img)
show_img(big_img)
show_img(small_img)
show_img(rotate_img)
show_img(binary_img)


# ## 3. 2枚の異なる画像の差分画像作成 

# In[4]:


# 画像の読み込み(グレースケール画像に変換する)
img1 = cv2.imread("Boolean_union.png", 0)
img2 = cv2.imread("Boolean_intersect.png", 0)

# 差分を取る(マイナスは0に置換)
sub_img = img1 - img2
sub_img = np.where(sub_img<0, 0, sub_img)

# 画像表示
show_img(img1)
show_img(img2)
show_img(sub_img)


# ## 4. 画像の特徴量抽出と図示 
# ## 4-1. ヒストグラム

# In[5]:



# 画像読み込み
photo = cv2.imread('photo.jpg')
# 画像表示
show_img(photo)
# ヒストグラム表示
color = ('b', 'g', 'r')
for i,col in enumerate(color):
    histr = cv2.calcHist([photo], [i], None, [256], [0,256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()
plt.close()

# ## 4-2. 特徴量抽出

# In[6]:


# SURFによる特徴量抽出
surf = cv2.xfeatures2d.SURF_create()
surf_kp = surf.detect(photo)
surf_img = cv2.drawKeypoints(photo, surf_kp, None, flags=4)

# A-KAZEによる特徴量抽出
akaze = cv2.AKAZE_create()
akaze_kp = akaze.detect(photo)
akaze_img = cv2.drawKeypoints(photo, akaze_kp, None, flags=4)

# 画像表示
show_img(surf_img)
show_img(akaze_img)


# ## 4-3. 特徴点のマッチング

# In[7]:


# 特徴点のマッチングのメソッド
# target: ターゲット画像
# input_img: 入力画像
# algo: 特徴量抽出のアルゴリズム
def feature_matching(target, input_img, algo):
    # 特徴量抽出
    kp1, des1 = algo.detectAndCompute(target, None)
    kp2, des2 = algo.detectAndCompute(input_img, None)
    
    # 総当たりマッチング
    bf = cv2.BFMatcher()
    # ターゲット画像の各キーポイントの特徴量に、最もマッチングした入力画像の上位k個の特徴量を返す
    matches = bf.knnMatch(des1, des2, k=2)
    
    ## ratio test
    # 1位の距離が2位の距離のratio以下であるものを採用、それ以外を間引き
    ratio = 0.6
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append([m])      
    print("{} -> {} points".format(len(matches), len(good)))
    
    # 出力
    output = cv2.drawMatchesKnn(target, kp1, input_img, kp2, good, None, flags=2)
    show_img(output)


# In[8]:


# 画像の読み込み
target = cv2.imread("target.JPG")
input_img = cv2.imread("input.JPG")

# SURFによる特徴マッチング
feature_matching(target, input_img, surf)
# A-KAZEによる特徴マッチング
feature_matching(target, input_img, akaze)

