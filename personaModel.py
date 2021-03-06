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

drop_rate = 0.3

class Encoder(chainer.Chain):
    def __init__(self, n_hidden):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.lstm_forward1 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_backward1 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_forward2 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_backward2 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_forward3 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_backward3 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_forward4 = L.StatelessLSTM(n_hidden, n_hidden)
            self.lstm_backward4 = L.StatelessLSTM(n_hidden, n_hidden)#wordとpersona
    
    def __call__(self, ex):
        #this function is called in training
        #(文字数、バッチ、文字の埋め込みサイズ)
        #c,h(batch, hidden) Variable
        #ex = (50, 64, 100) Variable
        batch_num = np.shape(ex)[1]
        hidden_num = np.shape(ex)[2]
        h = chainer.Variable(self.xp.zeros((batch_num, hidden_num), dtype=np.float32))
        c = chainer.Variable(self.xp.zeros((batch_num, hidden_num), dtype=np.float32))
        for f_word, b_word in zip(ex, reversed(ex)):
            c_forward, h_forward = self.lstm_forward1(c, h, f_word) #入力c, hがいる
            c_backward, h_backward = self.lstm_backward1(c, h, b_word)
            c = c_forward / 2 + c_backward / 2
            h = h_forward / 2 + h_backward / 2
            #h = F.dropout(h, ratio=drop_rate)

            c_forward, h_forward = self.lstm_forward2(c, h, f_word) #入力c, hがいる
            c_backward, h_backward = self.lstm_backward2(c, h, b_word)
            c = c_forward / 2 + c_backward / 2
            h = h_forward / 2 + h_backward / 2
            #h = F.dropout(h, ratio=drop_rate)

            c_forward, h_forward = self.lstm_forward3(c, h, f_word) #入力c, hがいる
            c_backward, h_backward = self.lstm_backward3(c, h, b_word)
            c = c_forward / 2 + c_backward / 2
            h = h_forward / 2 + h_backward / 2
            #h = F.dropout(h, ratio=drop_rate)

            c_forward, h_forward = self.lstm_forward4(c, h, f_word) #入力c, hがいる
            c_backward, h_backward = self.lstm_backward4(c, h, b_word)
            c = c_forward / 2 + c_backward / 2
            h = h_forward / 2 + h_backward / 2
            #h = F.dropout(h, ratio=drop_rate)
        return c, h

 

class Decoder(chainer.Chain):
    #lstm decoder
    def __init__(self, n_vocab, n_hidden, embed):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.lstmD1 = L.StatelessLSTM(n_hidden*2, n_hidden)#word+persona
            self.lstmD2 = L.StatelessLSTM(n_hidden*2, n_hidden)
            self.lstmD3 = L.StatelessLSTM(n_hidden*2, n_hidden)
            self.lstmD4 = L.StatelessLSTM(n_hidden*2, n_hidden)
            self.lineD = L.Linear(n_hidden, n_vocab)
        self.n_hidden = n_hidden
        self.embed = embed

    def __call__(self, firstEOS, y, pey, c_input, h_input):
        #this function is called in training 
        #y Variable
        loss = 0 #loss initialize
        c_old = c_input
        h_old = h_input
        max_len, batch = np.shape(y)
        firstEOS = F.concat((firstEOS, pey), axis=1)#あってるか？peyが(batch,1,n_hidden)か？
        #print(firstEOS)いけそう？
        #batch = np.shape(y)[1]
        #yはint?
        #yiとpyの形が同じはず
        for yi in y:            
            c_new, h_new = self.lstmD1(c_old, h_old, firstEOS)
            #h_new = F.dropout(h_new, ratio=drop_rate)
            c_new, h_new = self.lstmD2(c_new, h_new, firstEOS)
            #h_new = F.dropout(h_new, ratio=drop_rate)
            c_new, h_new = self.lstmD3(c_new, h_new, firstEOS)
            #h_new = F.dropout(h_new, ratio=drop_rate)
            c_new, h_new = self.lstmD4(c_new, h_new, firstEOS)
            py = self.lineD(h_new)#n_hidden => n_vocab
            #不安おそらくバッチ１ワードずつなはず？
            #yi = xp.array(y[i], dtype=xp.int32)
            loss += F.sum(F.softmax_cross_entropy(py, yi, reduce='no', ignore_label=-1))#intしか無理これはいるreduceはbatchsizeこlossが出る
            firstEOS = loadData.LoadData.sequence_embed(self.embed, yi.data)#memory削減
            firstEOS = F.concat((firstEOS, pey), axis=1)
            c_old = c_new
            h_old = c_new
            #print(loss) Nan
        return loss/batch


    def predict(self, firstEOS, targetPersona, c_input, h_input):
        #input_wordはembedding後を入れてくれeosもナ
        #returnはembedを返すナ
        h_old = h_input
        c_old = c_input
        firstEOS = F.concat((firstEOS, targetPersona), axis=1)
        output_word = []
        for _ in range(loadData.LoadData.maxlen):
            c_new, h_new = self.lstmD1(c_old, h_old, firstEOS)
            c_new, h_new = self.lstmD2(c_new, h_new, firstEOS)
            c_new, h_new = self.lstmD3(c_new, h_new, firstEOS)
            c_new, h_new = self.lstmD4(c_new, h_new, firstEOS)
            py = self.lineD(h_new)#n_hidden => n_vocab
            py = self.xp.argmax(py.data, axis=1).astype(self.xp.int32)
            output_word.append(py)
            firstEOS = loadData.LoadData.sequence_embed(self.embed, py)
            firstEOS = F.concat((firstEOS, targetPersona), axis=1)
            c_old = c_new
            h_old = c_new
        return output_word
        

class Model(chainer.Chain):
    def __init__(self, n_hidden, batch, device):
        super(Model, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(len(loadData.LoadData.vocabulary), n_hidden, ignore_label=-1)
            self.personaEmb = L.EmbedID(len(loadData.LoadData.peVocabulary), n_hidden, ignore_label=-1)
            #self.lineCED = L.Linear(n_hidden*2, n_hidden)
            #self.lineHED = L.Linear(n_hidden*2, n_hidden)
            self.enc = Encoder(n_hidden)
            self.dec = Decoder(len(loadData.LoadData.vocabulary), n_hidden, self.emb)

        self.n_hidden = n_hidden
        self.batch = batch
        self.n_vocab = len(loadData.LoadData.vocabulary)
        self.n_personaVocab = len(loadData.LoadData.peVocabulary) 
        self.device = device

    def __call__(self, ex, ey, pex, pey):
        #this function is called in training
        #LSTMに入れるならex,eyはVariable
        #peyは誰が答えるかpxは話し手今回はいらんか？
        y = ey.T#Variable
        ex = self.emb(ex.T) #転置
        pey = self.personaEmbed(pey)

        #encode
        c, h = self.enc(ex)
        #context
        #c = self.lineCED(c)
        #h = self.lineHED(h)
        #h = F.dropout(h, ratio=.2)
        #h = self.con(h)
        #decode
        #デコーダーの一番目にぶち込むやつ
        firstEOS = self.xp.ones(self.batch, dtype=self.xp.int32)
        firstEOS = self.emb(firstEOS)

        loss = self.dec(firstEOS, y, pey, c, h)
        showLoss = self.xp.mean(loss.data)
        chainer.report({'loss': showLoss}, self)
        perp = self.xp.exp(showLoss * self.batch / loadData.LoadData.maxlen)
        chainer.report({'perp': perp}, self)

        return loss

    def predict(self, ex, pey):
        #出力は
        #peyはlist (1)intであってほしかった
        #batchで入ってくることもある
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            predictBatch = np.shape(ex)[0]
            ex = self.emb(ex.T)
            if np.shape(pey) == ():
                pey = self.xp.reshape(self.xp.asarray(pey), (1))
            else:
                pey = self.xp.asarray(pey, dtype=self.xp.int32)
            pey = self.personaEmb(pey) #listで入ってくるとだめ
            #pey = self.personaEmbed(pey)#peyが(1,)やないとあかん？
            #encode
            c, h = self.enc(ex)
            #context
            #c = self.lineCED(c)
            #h = self.lineHED(h)
            #h = self.con(h)
            #decode
            firstEOS = self.xp.ones(predictBatch, dtype=self.xp.int32)
            firstEOS = self.emb(firstEOS)
            result = self.dec.predict(firstEOS, pey, c, h)

            result = self.xp.asarray(result, dtype=self.xp.int32).T
        return result


    def personaEmbed(self, pey):
        noneNumber = loadData.LoadData.peVocabulary.get("none")
        #xp = cuda.get_array_module(pey)
        pem = self.xp.asarray([],dtype=np.float32)
        flag = 0
        for p in pey:
            p = self.xp.reshape(p, (1))
            if flag == 0:
                flag = 1
                if p == noneNumber:#Noneならゼロ
                    #axis = 1でいいか？
                    pem = chainer.Variable(self.xp.asarray([self.xp.zeros(self.n_hidden, dtype=np.float32)], dtype=np.float32))
                else:
                    pem = self.personaEmb(p)
            else:        
                if p == noneNumber:#Noneならゼロ
                    #axis = 1でいいか？
                    pem = F.concat((pem, self.xp.asarray([self.xp.zeros(self.n_hidden, dtype=np.float32)], dtype=np.float32)), axis=0)
                else:
                    temp = self.personaEmb(self.xp.asarray(p, dtype=np.int32))
                    pem = F.concat((pem, temp), axis=0)

        return pem


    
    
        
