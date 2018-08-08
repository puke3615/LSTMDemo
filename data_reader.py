# coding=utf-8
from collections import Counter
import numpy as np
import jieba
import json
import os

PATH = './data'
DIVIDER = '\n\n\n'


def get_all_files(path=PATH, file_selector=lambda filepath: True, dir_selector=lambda dirpath: True, result=None):
    result = [] if result is None else result
    if not os.path.exists(path):
        raise Exception('File "%s" not found' % path)
    isfile = os.path.isfile(path)
    if isfile:
        if not file_selector or file_selector(path):
            result.append(path)
    else:
        if not dir_selector or dir_selector(path):
            for sub in os.listdir(path):
                sub_path = os.path.join(path, sub)
                get_all_files(sub_path, file_selector, dir_selector, result)
    return result


def read_file(path, result=None):
    result = set() if result is None else result
    with open(path, encoding='utf-8') as f:
        items = f.read().split(DIVIDER)
        for item in items:
            item = item.strip()
            if not item:
                continue
            properties = item.split('\n', 2)
            if not properties or len(properties) < 3:
                continue
            menu, star, text = [property[7:] for property in properties]
            result.add(text)
    return result


class CommentDataset:
    def __init__(self, path=PATH, n_words=2000, maxlen=30, log=False):
        self.path = path
        self.n_words = n_words
        self.maxlen = maxlen
        self.log = log

    def load_files(self):
        files = get_all_files(self.path, file_selector=lambda filepath: filepath.endswith('.txt'))
        # print '\n'.join(files)
        comments = set()
        for i, file in enumerate(files):
            if i == 3:
                break
            read_file(file, comments)
        comments = list(comments)
        print(len(comments))
        if self.log:
            print('\n'.join(comments))
        all_words = Counter()
        for comment in comments:
            for word in jieba.cut(comment):
                all_words[word] += 1
                # if log:
                #     print(word)
        total_words = len(all_words)
        self.all_words = all_words.most_common(self.n_words)
        print('Found %d words, select %d words.' % (total_words, len(self.all_words)))
        self.word2index = {word.encode('utf-8'): index for index, (word, times) in enumerate(self.all_words)}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.words = [word for word, times in self.all_words]
        self.data = []
        # 忽略词对应的值(同占位符)
        occupy_index = self.n_words
        for comment in comments:
            comment = comment.strip()
            if not comment:
                continue
            line = []
            for word in jieba.cut(comment):
                word = word.encode('utf-8')
                line.append(self.word2index[word] if word in self.word2index else occupy_index)
                # if log:
                #     print(word)
            self.data.append(line)

    def dump(self, filepath='data/final_data.json'):
        self.load_files()
        dump_data = {
            'words': self.words,
            'data': self.data
        }
        dir_path = os.path.dirname(os.path.abspath(filepath))
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        with open(filepath, 'w') as f:
            json.dump(dump_data, f, ensure_ascii=True)

    def load_data(self, reload=False, filepath='data/final_data.json'):
        if reload or not os.path.isfile(filepath):
            self.dump(filepath)
        with open(filepath) as f:
            json_data = json.load(f, encoding='utf-8')
        self.data = json_data['data']
        self.words = [word.encode('utf-8') for word in json_data['words']]
        if len(self.words) != self.n_words:
            raise Exception('Data error.')

        # 占位符
        v_padding = self.n_words
        # 结束符
        v_end = self.n_words + 1
        maxlen = self.maxlen

        from keras.preprocessing import sequence
        from keras.utils import np_utils
        x = []
        y = []
        for index, comment in enumerate(self.data):
            if self.log:
                print('%d/%d' % (index + 1, len(self.data)))
            for i, word in enumerate(comment):
                # if i % 3 != 0:
                #     continue
                input = comment[max(0, i - maxlen): i]
                output = comment[i + 1] if i < len(comment) - 1 else v_end
                x.append(input)
                y.append(output)
        # [self.to_words(v) for v in x[100:200]]
        X = sequence.pad_sequences(x, maxlen=maxlen, value=v_padding)
        Y = np_utils.to_categorical(y, self.n_words + 2)
        return X, Y

    def to_words(self, data):
        mapping = lambda i: self.words[i].decode('utf-8') if i < len(self.words) else 'å'
        return ''.join(map(mapping, data))

    def preprocess(self, standard=True):
        from keras.preprocessing import sequence
        result = sequence.pad_sequences(self.data, maxlen=self.maxlen, dtype=np.float32)
        if standard:
            result = result / self.n_words
        return result

    def print_words_times(self):
        for i, (k, v) in enumerate(self.all_words):
            print('%05d: %s: %s' % (i + 1, k, v))

    def summary(self):
        lengths = list(map(len, self.data))
        print("Max: %d" % np.max(lengths))
        print("Min: %d" % np.min(lengths))
        print("Mean: %d" % np.mean(lengths))
        print("Std: %d" % np.std(lengths))


if __name__ == '__main__':
    dataset = CommentDataset(log=True)
    # dataset.dump()
    data = dataset.load_data(reload=False)
    dataset.summary()
    print(1)
