"""

Usage

from Converter import make_dataset
make_dataset()

インポートしてmake_dataset()
ディレクトリの追加等は、make_dataset()内のpathsAndLabelsをいじってください
"""
import numpy as np
import codecs
import sys
import os
import chainer
from chainer import Variable
import matplotlib.pyplot as plt

class kihu(str):
    def __init__(self):
        self.x=np.zeros((11,9,9), dtype=np.float32)
        
    def convert(self, filename):
        super().__init__()
        x=self.x

        c=-1
        
        sys.stdin=codecs.open(filename, "r", "shift_jis")

        for line in sys.stdin:
            if line[0] == "|":
                c=c+1
                for i in range(0, 10):
                    if line[2*i]=="・":
                        x[0][i-1][c]=1
                    elif line[2*i]=="歩":
                        x[1][i-1][c]=1
                    elif line[2*i]=="金":
                        x[2][i-1][c]=1
                    elif line[2*i]=="銀":
                        x[3][i-1][c]=1
                    elif line[2*i]=="桂":
                        x[4][i-1][c]=1
                    elif line[2*i]=="香":
                        x[5][i-1][c]=1
                    elif line[2*i]=="飛":
                        x[6][i-1][c]=1
                    elif line[2*i]=="角":
                        x[7][i-1][c]=1
                    elif line[2*i]=="玉":
                        x[8][i-1][c]=1
                    elif line[2*i]=="と":
                        x[9][i-1][c]=1
                    elif line[2*i]=="龍":
                        x[10][i-1][c]=1
        
        return x
            
def make_dataset():
    # Directory name   "./DIRECTORY_NAME/"
    # 後ろのスラッシュは必須

    pathsAndLabels = []    # 読み込むディレクトリのリスト

    kihuData = []
    labelData = []

    for pathsAndLabel in pathsAndLabels:
        for (root, dirs, files) in os.walk(pathsAndLabel[0]):
            for file in files:
                k=kihu()
                name, ext = os.path.splitext(file)
                if ext == ".kif":
                    x = k.convert(pathsAndLabel[0]+file)
                    kihuData.append(x)
                    labelData.append(0)
                    
    # デバッグ用
    # print(kihuData)

    threshold = np.int32(len(kihuData)/10*9)    # threshold→トレーニングデータとテストデータを分ける閾値(値は適当)
                                                # 0~threshold トレーニングデータ
                                                # threshold~最後まで テストデータ
            
    train = chainer.datasets.tuple_dataset.TupleDataset(kihuData[0:threshold], labelData[0:threshold])
    test = chainer.datasets.tuple_dataset.TupleDataset(kihuData[threshold:], labelData[threshold:])
    return train, test    # train[#番目の盤面][0=盤面,1=ラベル][チャンネル][縦][横]
                           # testも同様

