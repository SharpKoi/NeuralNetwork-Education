import os
import cupy as cp
import numpy as np

from sproutnet.nn.model import Model
from sproutnet.nn.layers import Embedding, LSTM, Linear, Activation
from sproutnet.nn.activator import sigmoid
from sproutnet.nn.metric import BinaryAccuracy
from sproutnet.nn.loss import binary_cross_entropy

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(cp.cuda.runtime.getDevice())

n_data = 6400
n_vocab = 1000
text_len = 20
vec_dim = 100

W = cp.random.randn(n_vocab, vec_dim)
X = np.random.randint(0, n_vocab, size=(n_data, text_len))

WX = cp.empty(shape=(n_data, text_len, vec_dim))
for i in range(n_data):
    WX[i] = W[X[i]]
Y = np.array([[int(item >= 200)] for item in np.sum(np.linalg.norm(WX.get(), axis=-1), axis=-1)])

model = Model(loss=binary_cross_entropy)
emb_layer = Embedding(input_length=text_len, weights=W)
lstm_layer = LSTM(units=16)
linear_layer = Linear(units=1)
activation_layer = Activation(activator=sigmoid)
model.add_layer(emb_layer, lstm_layer, linear_layer, activation_layer)

model.fit(X, Y, 10, learning_rate=0.2, batch_size=64, metric=BinaryAccuracy())
y_pred = model.predict(X[:10])
print(y_pred)
