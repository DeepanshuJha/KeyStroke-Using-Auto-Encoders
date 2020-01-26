from data_loader import loader
from model import Models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import numpy as np
import sys

w = int(sys.argv[1])
h = 1
model = Models(h, w)
auto_encoder = model.encode_decoder()
model.encode_decoder()

load_data = loader()    
print(model.encode_decoder().summary())

x_data, y_data = load_data.load()
print(x_data.shape)
train_x,test_x,train_y,test_y=train_test_split(x_data,y_data,test_size=0.2,random_state=30)

model.fit(train_x, train_y, test_x, test_y)

model.save()

