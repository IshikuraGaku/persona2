#-*- coding: utf-8 -*-

#引数としてファイルパスを指定すること

import urllib.request
import re
import html
import traceback
import sys
import random

sum = ""

try:
    myInputFile = open("PInput.txt", 'r')
    myOutputFile = open("POutput.txt", 'r')
    myTestInputFile = open("PTestInput.txt", "w")
    myTestOutputFile = open("PTestOutput.txt", "w")
    newInputFile = open("PNewInput.txt", "w")
    newOutputFile = open("PNewOutput.txt", "w")
except Exception as e:
    print("input,outputText error")
    traceback.print_exc()

inputRead = myInputFile.read()
outputRead = myOutputFile.read()

#小文字に
inputRead = inputRead.lower()
outputRead = outputRead.lower()

inputRead = re.sub(r"\n[^ :\n]* : *\n", "", inputRead) # hoge : #を消す
outputRead = re.sub(r"\n[^ :\n]* : *\n", "", outputRead) # hoge : #を消す

inputLine = inputRead.split("\n")
outputLine = outputRead.split("\n")

fileLength = len(inputLine)
tempTestInput = ""
tempTestOutput = ""

assert(len(inputLine) == len(outputLine))

for i in range(50000):
    print(fileLength-i-1)
    randnum = random.randint(0,fileLength-i-1)
    print(randnum)
    tempTestInput += inputLine[randnum] + "\n"
    print(randnum)
    tempTestOutput += outputLine[randnum] + "\n"
    print(randnum)
    del inputLine[randnum]
    del outputLine[randnum]     

myTestInputFile.write(tempTestInput)
myTestOutputFile.write(tempTestOutput)

for line in inputLine:
    newInputFile.write(line+"\n")

for line in outputLine:
    newOutputFile.write(line+"\n")



myInputFile.close()
myOutputFile.close()
myTestInputFile.close()
myTestOutputFile.close()
newInputFile.close()
newOutputFile.close()

