# Use the trained Model
encoder = K.function([model.layers[0].input], [model.layers[4].output])
encoded_images = encoder([x_train])[0].reshape(-1,7*7*7)
cnn.printDataSet(y_train, restored_testing_dataset, x_train)

clustered_training_set = calculateKMeans(encoded_images,y_train)
printConfusionMatrix(clustered_training_set, y_train)
accuracy = calculateAccuracy(clustered_training_set, y_train)
print(accuracy)

from sklearn.mixture import GaussianMixture as GMM

class GMMModel():
    def __init__(self, n_components):
        self.mNCompnents = n_components
        
    def buildModel(self):
        gmm = GMM(n_components = self.mNCompnents)
        return gmm
    
gmm = GMMModel(10)
modelGmm = gmm.buildModel()
gmmEncoder = modelGmm.fit(encoded_images)
labels = gmmEncoder.predict(encoded_images)
printConfusionMatrix(labels, y_train)
accuracy = calculateAccuracy(clustered_training_set, y_train)
print(accuracy)
