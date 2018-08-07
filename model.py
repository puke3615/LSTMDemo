# coding=utf-8
from keras.preprocessing import sequence
from data_reader import CommentDataset
from keras.callbacks import *
from keras.models import *
from keras.layers import *
import jieba


def choose_result(outputs):
    # 默认选择最大概率的结果

    return np.argmax(outputs)


def choose_result_by_prob(predict):
    # 取词逻辑
    # 将predict累加求和
    t = np.cumsum(predict)
    # 求出预测可能性的总和
    s = np.sum(predict)
    # 返回将0~s的随机值插值到t中的索引值
    # 由于predict各维度对应的词向量是按照训练数据集的频率进行排序的
    # 故P(x|predict[i]均等时) > P(x + δ), 即达到了权衡优先取前者和高概率词向量的目的
    return int(np.searchsorted(t, np.random.rand(1) * s))


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

    def generate(self, words, starts=None):
        self.model.load_weights('data/model.h5')
        header = []
        if starts:
            items = [item.encode('utf-8') for item in jieba.cut(starts)]
            for item in items:
                if item not in words:
                    raise Exception('Words not contains % s' % item)
                header.append(words.index(item))
        result = header[:]
        input = sequence.pad_sequences([header], maxlen=maxlen, value=n_words)
        while True:
            outputs = self.model.predict(input)[0]
            output = choose_result_by_prob(outputs)
            if output == n_words:
                # 占位符重新生成
                continue
            if output == n_words + 1:
                # 结束符完毕
                break
            result.append(output)
            input = np.array([input[0][1:].tolist() + [output]])
        return result


n_words = 2000
maxlen = 30

if __name__ == '__main__':
    dataset = CommentDataset(n_words=n_words, maxlen=maxlen)
    X, Y = dataset.load_data()
    words = dataset.words
    # offset = 1000
    # X, Y = X[:offset], Y[:offset]

    model = CommentModel(n_words, seq_length=maxlen)
    # model.train(X, Y, batch_size=32, epochs=100, verbose=1)

    for _ in range(100):
        result = model.generate(words, '')
        generated = dataset.to_words(result)
        # print('\nGenerated:')
        print(generated)
