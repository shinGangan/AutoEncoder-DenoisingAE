#------------------------------------------------------------
#   coding:utf-8
#------------------------------------------------------------
#	Updata History
#	January  15  17:00, 2019 (Tue) by S.Iwamaru
#------------------------------------------------------------
#
#	Cifar10を用いてDAE(Denoising AutoEncoder)を作成
#	手順としては
#		1) ノイズ付与画像を生成
#		2) モデルにより学習、テスト
#
#------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

"""
	モデルの生成、学習
"""
def model( x_train_noisy, x_train, x_test_noisy, x_test ):
	input_img = Input( shape=( 32, 32, 3) )	#  32×32、RGB

	#  Encode_Conv1
	x = Conv2D( 32, (3, 3), padding='same' )(input_img)
	X = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = MaxPooling2D( (2, 2), padding='same' )(x)
	#  Encode_Conv2
	x = Conv2D( 32, (3, 3), padding='same' )(x)
	X = BatchNormalization()(x)
	x = Activation('relu')(x)
	encoded = MaxPooling2D( (2, 2), padding='same' )(x)

	#  Decode_Conv1
	x = Conv2D( 32, (3, 3), padding='same' )(encoded)
	X = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)
	#  Decode_Conv2
	x = Conv2D( 32, (3, 3), padding='same' )(x)
	X = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = UpSampling2D((2, 2))(x)
	#  Decode_Conv3
	x = Conv2D( 3, (3, 3), padding='same' )(x)
	X = BatchNormalization()(x)
	decoded = Activation('sigmoid')(x)

	#  conpile
	autoencoder = Model( input_img, decoded )
	autoencoder.compile( optimizer='adam',
	                     loss='binary_crossentropy',
	                     metrics=["accuracy"] )

	#  アーキテクチャの可視化
	autoencoder.summary()	#  ディスプレイ上に表示
	plot_model( autoencoder, to_file="architecture.png" )

	epochs = 200
	batch_size = 128

	tensor_board = TensorBoard( "./logs", histogram_freq=0, write_graph=True, write_images=True )	
#	check_point = ModelCheckpoint( filepath="./model/model.{epoch:02d}-{val_loss:.2f}.h5", 
#   				monitor="val_loss")

	history = autoencoder.fit( x_train_noisy, x_train,
				epochs=epochs,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(x_test_noisy,x_test),
				callbacks=[ tensor_board ] )
"""
	データの成形、グラフ化
"""
if __name__ == '__main__':
	#	Cifer10のLoad, データの分割
	num_classes = 10
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()	#  学習:50000, テスト:10000
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

#	x_validation = x_test[:7000]	#  validation_data : ( 7000, 32, 32, 3)
#	x_test = x_test[7000:]			#  test_data : ( 3000, 32, 32, 3)

	"""
		ノイズの付与
	"""
	noise_factor = 0.1

	#  平均0、標準偏差1の正規分布
	x_train_noisy = x_train + noise_factor * np.random.normal( loc=0., scale=1., size=x_train.shape )
	x_test_noisy = x_test + noise_factor * np.random.normal( loc=0., scale=1., size=x_test.shape )
#	x_validation_noisy = x_validation + noise_factor * np.random.normal( loc=0., scale=1., size=x_validation.shape )

	x_train_noisy = np.clip( x_train_noisy, 0., 1. )
	x_test_noisy = np.clip( x_test_noisy, 0., 1. )
#	x_validation_noisy = np.clip( x_validation_noisy, 0., 1. )

	#  モデル関数
	model( x_train_noisy, x_train, x_test_noisy, x_test )

	#	グラフへ可視化(画像の表示)
	n = 10
	plt.figure(figsize=( 20, 4))
	for i in range(n):
		#  ノイズ付与画像の表示
		ax = plt.subplot( 2, n, i+1 )
		plt.imshow( x_test_noisy[i].reshape( 32, 32, 3) )
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		#  変換された画像の表示
		ax = plt.subplot( 2, n, i+1+n )
		plt.imshow( x_test[i].reshape( 32, 32, 3) )
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()
