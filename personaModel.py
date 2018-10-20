import loadData

import functools
import operator
import six
import chainer
from chainer.functions.activation import lstm
from chainer.functions.array import concat
from chainer.functions.array import split_axis
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.backends import cuda
import numpy as np

def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs    

class PersonaModel(chainer.Chain):
    def __init__(self, n_layers, n_hidden, n_vocab):
        super(PersonaModel, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(len(loadData.LoadData.vocabulary), n_hidden, ignore_label=-1)
            self.personaEmb = L.EmbedID(len(loadData.LoadData.peVocabulary), n_hidden, ignore_label=-1)
            self.enc = L.NStepBiLSTM(n_layers, n_hidden, n_hidden, 0.3)
            self.dec = L.NStepLSTM(n_layers, n_hidden*2, n_hidden, 0.3)
            self.W = L.Linear(n_hidden, n_vocab)
        
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.n_vocab = n_vocab
        
    def __call__(self, x, y, px, py):
        #x
        ex = sequence_embed(self.emb, x)
        y_in = [F.concat([loadData.LoadData.EOS, temp], axis=0) for temp in y]#batchやから
        y_out = [F.concat([temp, loadData.LoadData.EOS], axis=0) for temp in y]
        ey_in = sequence_embed(self.emb, y_in)
        epy = self.personaEmb(py)
        batch = len(x)

        #encode
        h, c, _ = self.enc(None, None, ex)

        print("h")
        print(np.shape(h))
        print("c")
        print(np.shape(c))

        h = h[0] + h[1]
        c = c[0] + c[1]

        #decode
        #ToDo ey_inとepyを結合しろ
        ey_inp = words_persona_concat(ey_in, epy)
        _, _, prey = self.dec(h, c, ey_inp)

        prey = self.W(prey)
        prey = F.concat(prey, axis=0)
        concat_y_out = F.concat(y_out, axis=0)

        loss = F.sum(F.softmax_cross_entropy(
            prey, concat_y_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_y_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)

        return loss

        def translate(self, x, max_length=100):
            batch = len(x)
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                ex = sequence_embed(self.embed, x)
                h, c, _ = self.enc(None, None, ex)

                print("h")
                print(np.shape(h))
                print("c")
                print(np.shape(c))

                h = h[0] + h[1]
                c = c[0] + c[1]

                y = self.xp.full(batch, loadData.LoadData.EOS, dtype=np.int32)
                result = []
                for i in range(max_length):
                    ey = self.embed(y)
                    ey = F.split_axis(ey, batch, 0)
                    h, c, y = self.decoder(h, c, ey)
                    cy = F.concat(y, axis=0)
                    wy = self.W(cy)
                    y = self.xp.argmax(wy.data, axis=1).astype(np.int32)
                    result.append(y)
            result = cuda.to_cpu(
                self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)
            outs = []
            for y in result:
                ends = np.argwhere(y == loadData.LoadData.EOS)
                if len(ends) > 0:
                    y = y[:ends[0, 0]]
                outs.append(y)
            return outs

        def words_persona_concat(self, ex, epy):
            #ex (batch, 可変, n_hidden)
            #epy (batch, n_hidden)
            for i in range(np.shape(epy)[0]):
                exepy = self.xp.asarray([F.concat(word, epy[i]) for word in ex[i]])
            return exepy




