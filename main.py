from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist=keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train, x_test=x_train/255, x_test/255

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)


model = keras.Sequential([
    #first layer 32 filtre 3x3 filitre ReLU activation polling --> 2x2
    keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    #sec leyar 64 filtre "" "" ""
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    # flattern layer
    keras.layers.Flatten(),
    #fully connected layer 128 neron
    keras.layers.Dense(128,activation='relu'),

    keras.layers.Dense(64,activation='relu'),
    #output layer
    keras.layers.Dense(10,activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))
test_loss, test_acc=model.evaluate(x_test, y_test)
print(test_acc)

model.save("cnn_modelim.keras")
