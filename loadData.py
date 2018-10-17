import progressbar
import six
import datetime

import numpy as np
import chainer.functions as F
from chainer.backends import cuda
import re

class LoadData:

    UNK = 0
    EOS = 1
    PAD = -1
    #seqenceの長さのMaxとMin
    minlen = 1
    maxlen = 50
    vocabulary = {}
    peVocabulary = {}
    ids_words = {}

    @classmethod
    def word_ids(cls, voPath):
        #wordをidにするword_ids
        #辞書{文字:数字} 文字を入れたら数字が得られる
        with open(voPath) as f:
            # +2 for UNK and EOS
            #line.strip 改行空白除去
            vocabulary = {line.strip(): i + 2 for i, line in enumerate(f)}
        vocabulary['<UNK>'] = 0
        vocabulary['<EOS>'] = 1
        vocabulary['<PAD>'] = -1 #いる？
        return vocabulary


    @classmethod
    def ids_word(cls, vocabulary):
        #idをwordにする
        #辞書{数字:文字} 数字を入れたら文字が得られる
        ids_words = {i: w for w, i in vocabulary.items()}
        return ids_words


    @classmethod
    def load_data(cls, path):
        n_lines = cls.count_lines(path)
        bar = progressbar.ProgressBar()
        wordData = []
        #humanData = []
        count = 0
        print('loading...: %s' % path)
        with open(path) as f:
            for line in bar(f, max_value=n_lines):
                parson = re.match(r'[^(:|\n)]*:', line)#:を抜いた人の名前2単語以上ならどうしよ？
                line = re.sub(r'[^(:|\n)]*:', "", line)
                if parson != None:
                    parson = parson.group(0)[:-1].strip()
                else:
                    count += 1
                    parson = "none"
                words = line.strip().split()
                #numpyにせんとメモリが
                wordArray = np.array([cls.vocabulary.get(w, cls.UNK) for w in words], dtype=np.int32)
                parson = np.array(cls.peVocabulary.get(parson, cls.UNK), dtype=np.int32)
                wordData.append((parson, wordArray))
        print(count)
        return wordData

    @classmethod
    def count_lines(cls, path):
        #lineを数える
        with open(path) as f:
            return sum([1 for _ in f])

    @classmethod
    def sequence_embed(cls, embed, xs):
        #各文の長さ len(xs)はバッチサイズ
        if len(np.shape(xs)) != 1:
            x_len = [len(x) for x in xs]
            #cumsum指定された軸方向に足し合わされていった値を要素とする配列が返されます。
            #x_len[:-1]は一番最後を抜いたリスト
            x_section =np.cumsum(x_len[:-1])
            #xsはVariableかnumpy.ndarrayかcupy.ndarrayのタプル
            #concatは配列の結合axis=0は縦に結合行数が多くなる
            ex = embed(F.concat(xs, axis=0))
            #spritされてタプルが帰る結合したのを元に戻してそう
            exs = F.split_axis(ex, x_section, 0)
        else:
            exs = embed(xs)
        return exs

    @classmethod
    def makeVocab(cls, voPath, peVoPath):
        cls.vocabulary = cls.word_ids(voPath)
        cls.peVocabulary = cls.word_ids(peVoPath)
        cls.ids_words = cls.ids_word(cls.vocabulary)

    @classmethod
    def makeData(cls, inPath, outPath, voPath, peVoPath):
        #return [(source wordID0, target wordID0),(),()], {word:ID,..}, $0の分割版いる？
        #[np.array0, np.array1,...] ID
        #inData[(parson, wordArray), (), ()]
        inData = cls.load_data(inPath)
        outData = cls.load_data(outPath)
        assert len(inData) == len(outData)#ちょい厳しい
        train_data = [#write
                ((s[0], np.append(s[1], cls.EOS)), (t[0], np.append(t[1], cls.EOS)))
                for s, t in six.moves.zip(inData, outData)
                if (cls.minlen <= len(s[1])+1 <= cls.maxlen
                    and
                    cls.minlen <= len(t[1])+1 <= cls.maxlen)
            ]
        #おまけ
        print('[{}] Dataset loaded.'.format(datetime.datetime.now()))
        train_source_unknown = cls.calculate_unknown_ratio(
                [s[1] for s, _ in train_data])
        train_target_unknown = cls.calculate_unknown_ratio(
                [t[1] for _, t in train_data])

        print('vocabulary size: %d' % len(cls.vocabulary))
        print('persona vocabulary size %d' % len(cls.peVocabulary))
        print('Train data size: %d' % len(train_data))
        print('source unknown ratio: %.2f%%' % (
            train_source_unknown * 100))
        print('target unknown ratio: %.2f%%' % (
            train_target_unknown * 100))

        #train_data[((inparson, inArray),(outparson, outArray)), ((inparson, inArray),(outparson, outArray))]
        return (train_data, inData, outData)

    @classmethod
    def calculate_unknown_ratio(cls, data):
        unknown = sum((s == cls.UNK).sum() for s in data)
        total = sum(s.size for s in data)
        return unknown / total


