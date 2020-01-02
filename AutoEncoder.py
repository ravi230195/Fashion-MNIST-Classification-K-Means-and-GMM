class CNN():
    def __init__(self, input_shape):
        self.mInput_shape = input_shape

    def buildModel(self):
        model = Sequential()
        model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu', input_shape= self.mInput_shape))
        model.add(MaxPool2D((2,2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPool2D((2,2), padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(7, kernel_size=3, padding='same', activation='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(14, kernel_size=3, padding='same', activation='relu'))
        model.add(UpSampling2D((2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(1, kernel_size=3, padding='same', activation='relu'))
        return model
    
    def printDataSet(self, y_test, restored_testing_dataset, x_test):
        plt.figure(figsize=(20,5))
        for i in range(10):
            index = y_test.tolist().index(i)
            plt.subplot(2, 10, i+1)
            plt.imshow(x_test[index].reshape((28,28)))
            plt.gray()
            plt.subplot(2, 10, i+11)
            plt.imshow(restored_testing_dataset[index].reshape((28,28)))
            plt.gray()

# Build the autoencoder
input_shape = (28,28,1)
x_train = x_train.reshape(-1,28,28,1) / 255
x_test = x_test.reshape(-1,28,28,1) / 255
epoch = 5

cnn =  CNN(input_shape)
model = cnn.buildModel()
model.compile(optimizer='adam', loss="mse")
model.summary()
autoEncoder = model.fit(x_train, x_train, epochs=epoch, batch_size=128, validation_split=0.2, verbose=1)
restored_testing_dataset = model.predict(x_test)
print(restored_testing_dataset.shape)
cnn.printDataSet(y_test, restored_testing_dataset, x_test)
printLoss(autoEncoder, epoch)
