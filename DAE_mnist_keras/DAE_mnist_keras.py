#------------------------------------------------------------
#   coding: utf-8
#------------------------------------------------------------
# 	Updata History
# 	January  18  00:00, 2019 (Fri) by S.Iwamaru
#------------------------------------------------------------
#
#	MNISTを用いてDAEを作成
#
#------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.datasets import mnist

"""
	モデルの生成、学習
"""
def model( x_train_noisy, x_train, x_test_noisy, x_test ):

	input_img = Input( shape=( 28, 28, 1) )		#  28×28、グレースケール

	#  Encode_Conv1
	x = Conv2D( 32, (3, 3), padding='same', activation='relu' )(input_img)
	x = MaxPooling2D( ( 2, 2), padding='same' )(x)
	#  Encode_Conv2
	x = Conv2D( 32, (3, 3), padding='same', activation='relu' )(x)
	encoded = MaxPooling2D( ( 2, 2), padding='same' )(x)
	
	#  Decode_Conv1
	x = Conv2D( 32, (3, 3), padding='same', activation='relu' )(encoded)
	x = UpSampling2D(( 2, 2))(x)
	#  Decode_Conv2
	x = Conv2D( 32, (3, 3), padding='same', activation='relu' )(x)
	x = UpSampling2D(( 2, 2))(x)
	decoded = Conv2D( 1, (3, 3), padding='same', activation='sigmoid' )(x)
	
	autoencoder = Model( input_img, decoded )
	autoencoder.compile( optimizer='adam',
			     loss='binary_crossentropy',
			     metrics=["accuracy"] )
	
	#  アーキテクチャの可視化
	autoencoder.summary()	#  ディスプレイ上に表示
	plot_model( autoencoder, to_file="architecture.png")
	
	#	モデルの学習
	epochs = 10
	batch_size = 128
	
	tensor_board = TensorBoard( "./logs", histogram_freq=0, write_graph=True, write_images=True )	
#	check_point = ModelCheckpoint( filepath="./model/model.{epoch:02d}-{val_loss:.2f}.h5", 
#    							   monitor="val_loss")

	history = autoencoder.fit( x_train_noisy, x_train,
				epochs=epochs,
				batch_size=batch_size,
				shuffle=True,
				validation_data=(x_test_noisy,x_test),
				callbacks=[ tensor_board ] )

	#  学習のグラフ化
	plot_history( history, epochs )

"""
	精度、損失のグラフ
"""
def plot_history(history,epochs):
	#print(history.history.keys())
	
	#  精度の経過をプロット
	plt.figure()
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	plt.plot( range(epochs), acc, marker='.', label='acc' )
	plt.plot( range(epochs), val_acc, marker='.', label='val_acc' )
	plt.legend( loc='best', fontsize=10 )
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.title('model accuracy')
	plt.legend( ['acc','val_acc'], loc='lower right')
	plt.savefig('model_accuracy.png')
#	plt.show()

	
	#  損失の経過をプロット
	plt.figure()
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	plt.plot( range(epochs), loss, marker='.', label='loss' )
	plt.plot( range(epochs), val_loss, marker='.', label='val_loss' )
	plt.legend( loc='best', fontsize=10 )
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.title('model loss')
	plt.legend( ['loss','val_loss'], loc='lower right')
	plt.savefig('model_loss.png')
#	plt.show()

if __name__ == '__main__':
	#  データのロード・分割
	(x_train, y_train), (x_test, y_test) = mnist.load_data()	#  (60000, 28, 28), (10000, 28, 28)
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	x_train = np.reshape( x_train, (len(x_train), 28, 28, 1) )	#  (使用数,チャネル数,x,y)=(60000, 1, 28, 28)
	x_test = np.reshape( x_test, (len(x_test), 28, 28, 1) )

	"""
		ノイズの付与
	"""	
	noise_factor = 0.5
	
	#  平均0、標準偏差1の正規分布
	x_train_noisy = x_train + noise_factor * np.random.normal( loc=0., scale=1., size=x_train.shape )
	x_test_noisy = x_test + noise_factor * np.random.normal( loc=0., scale=1., size=x_test.shape )
	
	x_train_noisy = np.clip( x_train_noisy, 0., 1. )
	x_test_noisy = np.clip( x_test_noisy, 0., 1. )

	#  モデル
	model( x_train_noisy, x_train, x_test_noisy, x_test )

	"""
		グラフへ可視化(画像の描画)
	"""
	#  表示数
	n = 10
	plt.figure(figsize=( 20, 4))
	for i in range(n):
		#  ノイズ付与画像の表示
		ax = plt.subplot( 2, n, i+1 )
		plt.imshow( x_test_noisy[i].reshape( 28, 28) )
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	
		#  変換された画像の表示
		ax = plt.subplot( 2, n, i+1+n )
		plt.imshow( x_test[i].reshape( 28, 28) )
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

