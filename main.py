import numpy as np
np.set_printoptions(threshold=np.inf)
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

def generate_features(image_paths):
    # 画像データを格納する空の配列を初期化
    images = np.zeros(shape=(len(image_paths), 224, 224, 3))
    # 事前学習済みのVGG16モデルをロード
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
    # ペナルティメイト層の結果を出力するようモデルを修正
    model = Model(inputs=pretrained_vgg16.input, outputs=pretrained_vgg16.get_layer('fc2').output)
    
    # 各画像を読み込み、前処理を行う
    for i, f in enumerate(image_paths):
        img = image.load_img(f, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand
    # VGG16用に画像を前処理
    inputs = preprocess_input(images)
    
    # 前処理したデータを入力として処理し, ペナルティメイト層を出力する
    images_features = model.predict(inputs)
    return images_features

# 使用例
image_paths = ['./input.jpg']
features = generate_features(image_paths)
print(features)