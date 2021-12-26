import os
import cupy as cp
import numpy as np

from sproutnet.nn.model import Model
from sproutnet.nn.layers import Embedding, LSTM, Linear, Activation
from sproutnet.nn.activator import sigmoid
from sproutnet.nn.metric import BinaryAccuracy
from sproutnet.nn.loss import binary_cross_entropy
from sproutnet.nn.optimizer import Adam

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# print(f'CUDA available: {cp.cuda.is_available()}')
# print(f'Current CUDA device: {cp.cuda.device.get_device_id()}')

n_data = 6400
n_vocab = 1000
text_len = 20
vec_dim = 100

W = np.random.randn(n_vocab, vec_dim)
X = np.random.randint(0, n_vocab, size=(n_data, text_len))

WX = np.empty(shape=(n_data, text_len, vec_dim))
for i in range(n_data):
    WX[i] = W[X[i]]
Y = np.array([[int(item >= 200)] for item in np.sum(np.linalg.norm(WX, axis=-1), axis=-1)])

model = Model(loss=binary_cross_entropy, optimizer=Adam(), metric=BinaryAccuracy())
emb_layer = Embedding(input_length=text_len, weights=W)
lstm_layer = LSTM(units=32)
linear_layer = Linear(units=1)
activation_layer = Activation(activator=sigmoid, name='Sigmoid')
model.add_layer(emb_layer, lstm_layer, linear_layer, activation_layer)

print(model)
print(f'Total trainable params: {len(model.trainable_params)}')
for param in model.trainable_params:
    print(param.shape)
model.fit(X, Y, 10, batch_size=64)
# y_pred = model.predict(X[:10])
# print(y_pred)
