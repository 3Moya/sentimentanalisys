import pandas as pd
import numpy as np
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping

train = pd.read_csv('data/train.csv')
X_train = np.stack(train['feat_vector'].apply(eval))
y_train = np.array(pd.get_dummies((train['target']).values))

model = Sequential([
    Embedding(3000, 128, input_length=X_train.shape[1]),
    SpatialDropout1D(.2),
    LSTM(256, dropout=.8, recurrent_dropout=.8),
    Dense(2, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_accuracy', patience=3)
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=.2,
    shuffle=False,
    callbacks=[early_stop]
)

now = datetime.now()
date_time = now.strftime('%y%m%d%H%M%S')

model.save(f'model/model_{date_time}.h5')