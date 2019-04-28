#----------------------- -------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#	Updata History
#	December  06  16:00, 2018 (Thu) by S.Iwamaru
#------------------------------------------------------------
#
#	DeepAutoEncoder
#		https://blog.keras.io/building-autoencoders-in-keras.html
#------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.utils import plot_model

"""
	モデル生成
"""
input_img = Input(shape=(784,))

#  Encode部分
encoded = Dense(128, activation="relu")(input_img)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

#  Decode部分
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded = Dense(784, activation="sigmoid")(decoded)

#  Model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adadelta",
					loss="binary_crossentropy")

#  アーキテクチャの可視化
plot_model( autoencoder, to_file="architecture.png" )

"""
	データ読み込み
"""
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

epochs = 100
batch_size = 256

autoencoder.fit( x_train, x_train,
		epochs=epochs,
		batch_size=batch_size,
		shuffle=True,
		validation_data=(x_test, x_test))
				 
"""
	データの可視化
"""
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
	#  display original
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
plt.show()
