import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS,WordCloud
data_path="E:/datasets"
datasets=os.listdir(data_path)
for data in datasets:
    if data=="comment.csv":
        data_path=os.path.join(data_path,data)
data=pd.read_csv(data_path)
data.drop("id",axis=1,inplace=True)
features=list(data.columns)
y_features=[]
for feature in features:
    if type(data[feature][0]) != str:
        y_features.append(feature)
y_data=data[y_features]
x_data=data["comment_text"]
font={"size":8,"color":"red"}
class Data:

    def getx_data(self):
        return x_data
    def gety_data(self):
        return y_data
    def vizulaize_data(self):
        for idx, value in enumerate(y_features, start=1):
            plt.subplot(3, 3, idx)
            sns.countplot(x=value, data=data)
            plt.title(value, fontdict=font)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.show()


    def data_information(self,mydata):
        global columns
        columns = data.columns
        info = mydata.info()
        des = mydata.describe()
        # print(columns,info,des)

