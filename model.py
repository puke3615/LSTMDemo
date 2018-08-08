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
    def __init__(self, n_words, n_embeding=30, seq_length=30, save_dir='data/model.h5'):
        self.n_words = n_words
        self.final_words_dim = self.n_words + 2
        self.n_embeding = n_embeding
        self.seq_length = seq_length
        self.save_dir = save_dir
        dir = os.path.dirname(os.path.abspath(save_dir))
        if not os.path.isdir(dir):
            os.makedirs(dir)
        self._build_model()

    def _build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.final_words_dim, self.n_embeding, input_length=self.seq_length))
        self.model.add(LSTM(1024, return_sequences=True))
        self.model.add(LSTM(1024))
        self.model.add(Dense(2048))
        self.model.add(Dense(self.final_words_dim, activation='softmax'))
        self.model.summary()
        self.model.compile('adam', 'categorical_crossentropy', ['acc'])

    def train(self, X, Y, **kwargs):
        self.model.fit(X, Y, callbacks=[
            ModelCheckpoint(self.save_dir),
            TensorBoard(),
        ], **kwargs)

    def generate(self, words, starts=None):
        self.model.load_weights(self.save_dir)
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
maxlen = 20
save_dir = 'data/model.h5'
# train = True
train = False

if __name__ == '__main__':
    dataset = CommentDataset(n_words=n_words, maxlen=maxlen, log=False)
    X, Y = dataset.load_data()
    words = dataset.words
    # offset = 1000
    # X, Y = X[:offset], Y[:offset]

    model = CommentModel(n_words, seq_length=maxlen, save_dir=save_dir)

    if train:
        model.train(X, Y, batch_size=128, epochs=1000, verbose=1)
    else:
        for _ in range(100):
            result = model.generate(words, '')
            generated = dataset.to_words(result)
            # print('\nGenerated:')
            print(generated)
