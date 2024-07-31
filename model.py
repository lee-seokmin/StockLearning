from keras import Sequential
from keras.layers import Dense, LSTM


class Model():
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.action_dim = 2

    def model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(loss='mse', optimizer='adam')

        return model
