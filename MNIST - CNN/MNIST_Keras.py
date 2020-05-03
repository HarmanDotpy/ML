import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense , Conv2D, Dropout, Flatten,MaxPooling2D


(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
# image_index = 7777
# print(y_train[image_index])
# plt.imshow(x_train[image_index], cmap='Greys')
# plt.show()
# print(x_train.shape)

# 4d array to work with keras
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalizing
x_train /=255
x_test /=255

print('x_train_shape',x_train.shape)
print('number of images in x_train',x_train.shape[0])
print('number of images in x_test',x_test.shape[0])

model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=10)


print(model.evaluate(x_test, y_test))
