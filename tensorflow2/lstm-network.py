import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from sklearn import model_selection

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def lstm_model(X_train):
    model = tf.keras.Sequential(
        [
            layers.LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.Dense(units=1)
        ]
    )

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(learning_rate=0.001)
    )
    return model


if __name__ == '__main__':
    time_steps = 10

    # Generate data
    time = np.arange(0, 100, 0.1)
    sin = np.sin(time) + np.random.normal(scale=0.5, size=len(time))

    # data preprocessing
    df = pd.DataFrame(dict(sine=sin), index=time, columns=['sine'])
    train, test = model_selection.train_test_split(df, train_size=0.8)

    X_train, y_train = create_dataset(train, train.sine, time_steps)
    X_test, y_test = create_dataset(test, test.sine, time_steps)

    # build model
    model = lstm_model(X_train)
    print(model.summary())

    # training
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    # Evaluation
    y_pred = model.predict(X_test)