#!/usr/bin/env python

# -*- coding:utf-8 -*-

#python mySeq2seq2.0.py Input.preprocess.en Output.preprocess.en vocab.en vocab.en --validation-source testInput.preprocess.en --validation-target testOutput.preprocess.en --layer 3 --epoch 20 --unit 250 --save "test"

import argparse
import datetime

from nltk.translate import bleu_score
import numpy
import progressbar
import six

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    #各文の長さ len(xs)はバッチサイズ
    x_len = [len(x) for x in xs]
    #cumsum指定された軸方向に足し合わされていった値を要素とする配列が返されます。
    #x_len[:-1]は一番最後を抜いたリスト
    x_section = numpy.cumsum(x_len[:-1])
    #xsはVariableかnumpy.ndarrayかcupy.ndarrayのタプル
    #concatは配列の結合axis=0は縦に結合行数が多くなる
    ex = embed(F.concat(xs, axis=0))
    #spritされてタプルが帰る結合したのを元に戻してそう
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):
    #model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
    #source_ids = load_vocabulary(args.SOURCE_VOCAB)
    #target_ids = load_vocabulary(args.TARGET_VOCAB)
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            self.f_encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.2)
            self.b_encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.2)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.2)
            self.W = L.Linear(n_units, n_target_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    #The __call__ method takes sequences of source language’s word IDs xs and sequences of target language’s word IDs ys. 
    #Each sequence represents a sentence, and the size of xs is mini-batch size
    def __call__(self, xs, ys):
        #xを逆順に反転、xsは単語IDs
        xs = [x[::-1] for x in xs]
        #xpはnumpyかcupyを返す
        eos = self.xp.array([EOS], numpy.int32)
        #文の先頭にeosつける文字は数字
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        #埋め込みベクトルに
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        #__call__の引数に隠れ層の初期状態、セルの初期状態を渡す
        # NStepLSTM.__call__の第4引数xsは、(seq_size,n_in)のサイズを持つtupleの配列である。
        # ここでseq_sizeはひとつのシーケンスの長さ（=L)、n_inは入力データの次元である。
        # 配列の要素数はバッチサイズに相当する
        fhx, fcx, _ = self.f_encoder(None, None, exs)
        bhx, bcx, _ = self.b_encoder(None, None, exs)
        hx = (fhx + bhx) / 2.0 #hx.grad = None
        cx = (fcx + bcx) / 2.0
        #hx:hidden states cx:cell states ys:list of :class:~chainer.Variable
        #ys[t]: shape (L_t, N) L_t length of sequence for time t N: size of hidden unit
        _, _, os = self.decoder(hx, cx, eys) #os(64,長さ,250)
        print(self.W.b.grad)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        print(numpy.shape(self.W(concat_os)))
        print(numpy.shape(concat_ys_out))
        print(numpy.shape(concat_ys_out[0]))
        
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch #W(concat_os)=>(892,40002) concat_ys_out(892)

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss #Variable gradはNone?!

    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            #逆順にする
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            fh, fc, _ = self.f_encoder(None, None, exs)
            bh, bc, _ = self.b_encoder(None, None, exs)
            h = (fh + bh) / 2.0
            c = (fc + bc) / 2.0
            ys = self.xp.full(batch, EOS, numpy.int32)
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype(numpy.int32)
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x)
                                     for x in batch[:-1]], dtype=numpy.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                references.extend([[t.tolist()] for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                ys = [y.tolist()
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)
        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key: bleu})


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK)
                                 for w in words], numpy.int32)
            data.append(array)
    return data


def load_data_using_dataset_api(
        src_vocab, src_path, target_vocab, target_path, filter_func):

    def _transform_line(vocabulary, line):
        words = line.strip().split()
        return numpy.array(
            [vocabulary.get(w, UNK) for w in words], numpy.int32)

    def _transform(example):
        source, target = example
        return (
            _transform_line(src_vocab, source),
            _transform_line(target_vocab, target)
        )

    return chainer.datasets.TransformDataset(
        chainer.datasets.TextDataset(
            [src_path, target_path],
            encoding='utf-8',
            filter_func=filter_func
        ), _transform)


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')#defalt=1024
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--use-dataset-api', default=False,
                        action='store_true',
                        help='use TextDataset API to reduce CPU memory usage')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='number of iteration to show log')#default200
    parser.add_argument('--validation-interval', type=int, default=50,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')#default4000 validation検証
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--save', '-s', default='mySeq2seq2.0.model',
                        help='save model param file name')
    parser.add_argument("--test", "-t", default="null", 
                        help="test made model file path")
    parser.add_argument("--use", "-use", default="null", 
                        help="use made model file path")
    args = parser.parse_args()

    # Load pre-processed dataset
    print('[{}] Loading dataset... (this may take several minutes)'.format(
        datetime.datetime.now()))
    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    if args.use_dataset_api:
        # By using TextDataset, you can avoid loading whole dataset on memory.
        # This significantly reduces the host memory usage.
        def _filter_func(s, t):
            sl = len(s.strip().split())  # number of words in source line
            tl = len(t.strip().split())  # number of words in target line
            return (
                args.min_source_sentence <= sl <= args.max_source_sentence and
                args.min_target_sentence <= tl <= args.max_target_sentence)

        train_data = load_data_using_dataset_api(
            source_ids, args.SOURCE,
            target_ids, args.TARGET,
            _filter_func,
        )
    else:
        # Load all records on memory.
        train_source = load_data(source_ids, args.SOURCE)
        train_target = load_data(target_ids, args.TARGET)
        assert len(train_source) == len(train_target)

        train_data = [
            (s, t)
            for s, t in six.moves.zip(train_source, train_target)
            if (args.min_source_sentence <= len(s) <= args.max_source_sentence
                and
                args.min_target_sentence <= len(t) <= args.max_target_sentence)
        ]
    print('[{}] Dataset loaded.'.format(datetime.datetime.now()))

    if not args.use_dataset_api:
        # Skip printing statistics when using TextDataset API, as it is slow.
        train_source_unknown = calculate_unknown_ratio(
            [s for s, _ in train_data])
        train_target_unknown = calculate_unknown_ratio(
            [t for _, t in train_data])

        print('Source vocabulary size: %d' % len(source_ids))
        print('Target vocabulary size: %d' % len(target_ids))
        print('Train data size: %d' % len(train_data))
        print('Train source unknown ratio: %.2f%%' % (
            train_source_unknown * 100))
        print('Train target unknown ratio: %.2f%%' % (
            train_target_unknown * 100))

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}


    if args.test == "null":
        # Setup model
        model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)

        if args.use != "null":
            chainer.serializers.load_npz(args.use, model)
        
        if args.gpu >= 0:
            chainer.backends.cuda.get_device(args.gpu).use()
            model.to_gpu(args.gpu)

        # Setup optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        # Setup iterator
        train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

        # Setup updater and trainer
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer, converter=convert, device=args.gpu)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
        trainer.extend(extensions.LogReport(
            trigger=(args.log_interval, 'iteration')))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/perp', 'validation/main/perp', 'validation/main/bleu',
            'elapsed_time']),
            trigger=(args.log_interval, 'iteration'))

        if args.validation_source and args.validation_target:
            test_source = load_data(source_ids, args.validation_source)
            test_target = load_data(target_ids, args.validation_target)
            assert len(test_source) == len(test_target)
            test_data = list(six.moves.zip(test_source, test_target))
            test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
            test_source_unknown = calculate_unknown_ratio(
                [s for s, _ in test_data])
            test_target_unknown = calculate_unknown_ratio(
                [t for _, t in test_data])

            print('Validation data: %d' % len(test_data))
            print('Validation source unknown ratio: %.2f%%' %
                (test_source_unknown * 100))
            print('Validation target unknown ratio: %.2f%%' %
                (test_target_unknown * 100))

            @chainer.training.make_extension()
            def translate(trainer):
                source, target = test_data[numpy.random.choice(len(test_data))]
                result = model.translate([model.xp.array(source)])[0]

                source_sentence = ' '.join([source_words[x] for x in source])
                target_sentence = ' '.join([target_words[y] for y in target])
                result_sentence = ' '.join([target_words[y] for y in result])
                print('# source : ' + source_sentence)
                print('# result : ' + result_sentence)
                print('# expect : ' + target_sentence)

            trainer.extend(
                translate, trigger=(args.validation_interval, 'iteration'))
            trainer.extend(
                CalculateBleu(
                    model, test_data, 'validation/main/bleu', device=args.gpu),
                trigger=(args.validation_interval, 'iteration'))

        print('start training')
        trainer.run()
        chainer.serializers.save_npz(args.out+"/"+args.save, model)

    else:
        model = Seq2seq(args.layer, len(source_ids), len(target_ids), args.unit)
        chainer.serializers.load_npz(args.test, model)
        while(True):
            inputString = input()
            words = inputString.strip().split()
            array = numpy.array([source_ids.get(w, UNK)
                                 for w in words], numpy.int32)
            result = model.translate(model.xp.array([array]))[0]
            result_sentence = ' '.join([target_words[y] for y in result])
            print(result_sentence)
if __name__ == '__main__':
    main()
