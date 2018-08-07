from data_reader import CommentDataset
from keras.callbacks import *
from keras.models import *
from keras.layers import *


class CommentModel:
    def __init__(self, n_words, n_embeding=50, seq_length=30):
        self.n_words = n_words
        self.final_words_dim = self.n_words + 2
        self.n_embeding = n_embeding
        self.seq_length = seq_length
        self._build_model()

    def _build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.final_words_dim, self.n_embeding, input_length=self.seq_length))
        self.model.add(LSTM(256))
        self.model.add(Dense(1024))
        self.model.add(Dense(self.final_words_dim, activation='softmax'))
        self.model.summary()
        self.model.compile('adam', 'categorical_crossentropy', ['acc'])

    def train(self, X, Y, **kwargs):
        self.model.fit(X, Y, callbacks=[
            ModelCheckpoint('data/model.h5'),
            TensorBoard(),
        ], **kwargs)


n_words = 2000
maxlen = 30

if __name__ == '__main__':
    dataset = CommentDataset(n_words=n_words, maxlen=maxlen)
    X, Y = dataset.load_data()
    offset = 1000
    X, Y = X[:offset], Y[:offset]

    model = CommentModel(n_words, seq_length=maxlen)
    model.train(X, Y, batch_size=32, epochs=100, verbose=1)
