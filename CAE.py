#------------------------------------------------------------
#   coding: utf-8
#------------------------------------------------------------
# Updata History
# May  09  14:00, 2018 (Wed) by S.Iwamaru
#------------------------------------------------------------
#
#	CAE
#		https://blog.keras.io/building-autoencoders-in-keras.html
#
#------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import TensorBoard

"""
	モデルの生成
"""
input_img = Input(shape=( 28, 28, 1))

#  Encode_Conv1
x = Conv2D( 16, (3, 3), activation='relu', padding='same' )(input_img)
x = MaxPooling2D(( 2, 2), padding='same' )(x)
#  Encode_Conv2
x = Conv2D( 8, (3, 3), activation='relu', padding='same' )(x)
x = MaxPooling2D(( 2, 2), padding='same' )(x)
#  Encode_Conv3
x = Conv2D( 8, (3, 3), activation='relu', padding='same' )(x)
encoded = MaxPooling2D(( 2, 2), padding='same' )(x)

#  Decode_Conv1
x = Conv2D( 8, (3, 3), activation='relu', padding='same' )(encoded)
x = UpSampling2D(( 2, 2))(x)
#  Decode_Conv2
x = Conv2D( 8, (3, 3), activation='relu', padding='same' )(x)
x = UpSampling2D(( 2, 2))(x)
#  Decode_Conv3
x = Conv2D( 16, (3, 3), activation='relu' )(x)
x = UpSampling2D(( 2, 2))(x)
#  Decode_Conv4
decoded = Conv2D( 1, (3, 3), activation='sigmoid', padding='same' )(x)

autoencoder = Model( input_img, decoded )
autoencoder.compile( optimizer='adadelta', loss='binary_crossentropy' )

#  アーキテクチャの可視化
plot_model( autoencoder, to_file="architecture.png" )

"""
	データの読み込み
"""
(x_train, _), (x_test, _) = mnist.load_data()    #  (60000, 28, 28)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape( x_train, (len(x_train), 28, 28, 1) )	
x_test = np.reshape( x_test, (len(x_test), 28, 28, 1) )

epochs = 50
batch_size = 128

autoencoder.fit( x_train, x_train,
				 epochs=epochs,
				 batch_size=batch_size,
				 shuffle=True,
				 validation_data=(x_test,x_test),
				 callbacks=[TensorBoard(log_dir='./autoencoder')] )

"""
	グラフへ可視化
"""
decoded_imgs = autoencoder.predict(x_test)

#  表示数
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
