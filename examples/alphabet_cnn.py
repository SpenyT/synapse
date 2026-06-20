import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from synapse.utils import kaggle_download_csv
#from matplotlib import pyplot as plt

# import kaggle dataset & randomize
df = kaggle_download_csv("ashishguptajiit/handwritten-az")
df = shuffle(df, random_state=42).reset_index(drop=True)

labels = df.iloc[:, 0].to_numpy(dtype=int)
pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0

testing_size = 1000
training_size =  10000 #for now select first 20,000 rows

X_dev, Y_dev = pixels[:testing_size], labels[:testing_size]
X_train, Y_train = pixels[testing_size:training_size], labels[testing_size:training_size]

# plt.imshow(X_dev[1].reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()

from synapse.models import NeuralNetwork
from synapse.layers import FCL

model = [
    FCL(784, 142, activation='relu',  weight_initializer="he"),
    FCL(142, 142, activation='relu',  weight_initializer="he"),
    FCL(142,  26, activation=None)
]
net = NeuralNetwork(model, loss='cross_entropy')
net.train(X_train, Y_train, epochs=5, learning_rate=0.01)
test_acc = net.evaluate(X_dev, Y_dev)
print(f"Test accuracy: {test_acc:.4f}")

# visualize samples
from synapse.utils import show_random_predictions
show_random_predictions(net, X_dev, Y_dev, rows=3, cols=6, as_letter=True)

# test with completely new data set
print("\nDownloading new dataset...")
df2 = kaggle_download_csv("sachinpatel21/az-handwritten-alphabets-in-csv-format")
df2 = shuffle(df2, random_state=42).reset_index(drop=True)

labels_new = df2.iloc[:, 0].to_numpy(dtype=int)                     # (N,)
pixels_new = df2.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0     # (N, 784)
X_new, Y_new = pixels_new, labels_new
print("Download complete!")

# plt.imshow(X_new[1].reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()

print("\nEvaluating Accuracy with new dataset...")
test_acc = net.evaluate(X_new, Y_new)
print(f"Test Accuracy with new dataset: {test_acc:.4f}")