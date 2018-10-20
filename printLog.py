import pandas
import io
import os
import re
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

with open("log") as f:
    logData = json.load(f)
    logData = pandas.DataFrame(logData)
    logData = logData[["main/loss", "validation/main/bleu"]]
    print(logData)
    plt.figure()
    logData.plot(subplots=True)
    plt.savefig("image")
    plt.close()

    
  