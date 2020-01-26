from keras.models import load_model
import numpy as np

encoder = load_model(r'./weights/encoder_weights.h5')
decoder = load_model(r'./weights/decoder_weights.h5')

inputs = np.array([[0.2721247673034668,0.3040194511413574,0.4640316963195801,0.07986283302307129,0.09597635269165039,0.0783681869506836,0.14392399787902832]])
x = encoder.predict(inputs)
y = decoder.predict(x)

print('Input: {}'.format(inputs))
print('Encoded: {}'.format(x))
print('Decoded: {}'.format(y))