#python train.py Input.preprocess.en Output.preprocess.en vocab.en personaVocab.en --validation-source testInput.preprocess.en --validation-target testOutput.preprocess.en --unit 100 --gpu 0 --epoch 20
#python train.py PTestInput.en PTestOutput.en vocab.en personaVocab.en --validation-source PTestInput.en --validation-target PTestOutput.en --unit 3 --epoch 1 --batch 2

#python train.py PNewInput.en PNewOutput.en vocab.en persona_vocab.en --validation-source PTestInput.en --validation-target PTestOutput.en --unit 400 --epoch 20 --batch 230 --gpu 0 --save 10_17_4_400.model
import loadData
import personaModel
import argparse
from nltk.translate import bleu_score
import chainer
from chainer import training
from chainer.training import extensions
import numpy as np
from chainer.backends import cuda
import chainer.functions as F
import random

UNK = 0
EOS = 1
PAD = -1

def main():
    parser = argparse.ArgumentParser(description='Chainer persona')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('VOCAB', help='vocabulary file')
    parser.add_argument('PERSONA_VOCAB', help='persona vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batch', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')#defalt=1024
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='number of iteration to show log')#default200
    parser.add_argument('--validation_interval', type=int, default=50,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')#default4000 validation検証
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--save', '-s', default='persona.model',
                        help='save model param file name')
    parser.add_argument('--use', '-use', default='null', 
                        help='use made model file path')
    args = parser.parse_args()

    #[(source wordID0, target wordID0),(),()], vocabulary{'word':ID,...}
    loadData.LoadData.makeVocab(args.VOCAB, args.PERSONA_VOCAB)

    train_data, _, _ = loadData.LoadData.makeData(args.SOURCE, args.TARGET, args.VOCAB, args.PERSONA_VOCAB)#np

    model = personaModel.Model(args.unit, args.batch, args.gpu)
    
    if args.use != 'null':
        chainer.serializers.load_npz(args.use, model)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    #Setup potimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    #Setup iterator
    train_iter = chainer.iterators.SerialIterator(train_data, args.batch, shuffle=False)

    #setup  updater and trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)#convert 
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss', 
        'main/perp', 'balidation/main/perp', 'validation/main/bleu', 'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))

    if args.validation_source and args.validation_target:
        test_data, _, _ = loadData.LoadData.makeData(args.validation_source, args.validation_target, args.VOCAB, args.PERSONA_VOCAB)
        
        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[np.random.choice(len(test_data))]
            #sourceは(persona, sequence)
            if args.gpu >= 0:
                source = cuda.cupy.asarray(source[1], dtype=np.int32)
            else:
                source = source[1]
            print("source\n"+' '.join([loadData.LoadData.ids_words.get(int(y), UNK) for y in source]))
            source = F.pad_sequence([source], 50, -1)
            result = model.predict(source.data, target[0])
            #あってんのか？
            result = ' '.join([loadData.LoadData.ids_words.get(int(y), UNK) for y in result[0]])#ここは(1,50)しかないはず？
            print('\nresult')
            print(result)
            print("\ntrue\n"+' '.join([loadData.LoadData.ids_words.get(int(y), UNK) for y in target[1]]))
        
        trainer.extend(translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, 'validation/main/bleu', batch=args.batch, device=args.gpu), 
            trigger=(args.validation_interval, 'iteration'))
        
        print('start training')
        trainer.run()
        chainer.serializers.save_npz(args.out+'/'+args.save, model)

#配列のタプルをタプルの配列に変換　trainerで使う
def convert(batch, device):
    if device >= 0:
        paddingX = [cuda.cupy.asarray(x[1]) for x, _ in batch]
        paddingY = [cuda.cupy.asarray(y[1]) for _, y in batch]
        personaX = [cuda.cupy.asarray(x[0]) for x, _ in batch]
        personaY = [cuda.cupy.asarray(y[0]) for _, y in batch]
    else:
        paddingX = [x[1] for x, _ in batch]
        paddingY = [y[1] for _, y in batch]
        personaX = [x[0] for x, _ in batch]
        personaY = [y[0] for _, y in batch]
    
    paddingX = F.pad_sequence(paddingX, 50, -1)
    paddingY = F.pad_sequence(paddingY, 50, -1)#Variable


    #numpyからVariableになったpadding, personaは?多分list,xp
    return {'ex': paddingX, 'ey': paddingY,
            'pex': personaX, 'pey': personaY}

class CalculateBleu(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=64, device=-1, max_length=50):
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

            use_data = random.sample(self.test_data, self.batch)

            if self.device >= 0:
                sources = [cuda.cupy.asarray(x[1]) for x, _ in use_data]
                targets = [cuda.cupy.asarray(y[1]) for _, y in use_data]
                #sourcePersona = [cuda.cupy.asarray(x[0]) for x, _ in use_data]#今は使わん将来使うかも？
                targetPersona = [cuda.cupy.asarray(y[0]) for _, y in use_data]
            else:
                sources = [x[1] for x, _ in use_data]
                targets = [y[1] for _, y in use_data]
                sourcePersona = [x[0] for x, _ in use_data]
                targetPersona = [y[0] for _, y in use_data]
            
            sources = F.pad_sequence(sources, loadData.LoadData.maxlen, -1)
            targets = F.pad_sequence(targets, loadData.LoadData.maxlen, -1)

            references.extend([[t.tolist()] for t in targets.data])
            ys = [y.tolist()
                for y in self.model.predict(sources.data, targetPersona)]#batch_size
            hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)
        chainer.report({self.key:bleu})
    

if __name__ == '__main__':
    main()


