#python test.py vocab.en personaVocab.en --batch 50 --gpu 0 --unit 300 --use persona.model
#python test.py vocab.en person_vocab.en --batch 50 --gpu 0 --unit 400 --use 4_400.model


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
import collections
import io
import re


def processingSentence(useSentence):
    split_pattern = re.compile(r'([.,\-\…“‘”/!?"\':;)(—´\{\}\*\#$\>\<])')
    useSentence = useSentence.lower()
    words = []
    for word in useSentence.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words




def main():
    parser = argparse.ArgumentParser(description='Chainer persona test')
    parser.add_argument('VOCAB', help='vocabulary file')
    parser.add_argument('PERSONA_VOCAB', help='persona vocabulary file')
    parser.add_argument('--batch', '-b', type=int, default=64,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='number of units')#defalt=1024
    parser.add_argument('--min-soupirce-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=1,
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    parser.add_argument('--use', '-use', default='null', 
                        help='use made model file path')
    args = parser.parse_args()
    
    loadData.LoadData.makeVocab(args.VOCAB, args.PERSONA_VOCAB)
    model = personaModel.Model(args.unit, args.batch, args.gpu)
    chainer.serializers.load_npz(args.out+"/"+args.use, model)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    print("input persona")
    persona = input()
    persona = persona.lower()
    persona = loadData.LoadData.peVocabulary.get(persona, loadData.LoadData.peVocabulary.get("none"))
    print(persona)

    while(True):
        print("input sentence")
        inputSentence = input()
        inputSentence = inputSentence.lower()
        words = processingSentence(inputSentence)
        idsArray = [loadData.LoadData.vocabulary.get(w, loadData.LoadData.UNK) for w in words]
        idsArray.append(loadData.LoadData.EOS)
        print(idsArray)
        idsArray = model.xp.asarray([idsArray], dtype=model.xp.int32)
        result = model.predict(idsArray, persona)
        result = ' '.join([loadData.LoadData.ids_words.get(int(w), loadData.LoadData.UNK) for w in result[0]])
        print("output")
        print(result+"\n")

if __name__ == "__main__":
    main()


