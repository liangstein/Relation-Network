import os;
import sys;
import json;
import pickle;
import numpy as np;
from PIL import Image;
from keras.models import Model;
from keras import backend as K;
from keras.layers.embeddings import Embedding;
from keras.utils.vis_utils import plot_model;
from keras.models import Sequential, load_model;
from keras.optimizers import rmsprop, adam, adagrad, SGD;
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau;
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer;
from keras.layers import Input, Dense, merge, Dropout, LSTM, BatchNormalization, \
    Activation, Conv2D, Lambda, Concatenate, Add;
DIR=os.getcwd();
with open(DIR+"/word_index_q","rb") as f:
    word_index_q=pickle.load(f);

with open(DIR+"/word_index_a","rb") as f:
    word_index_a=pickle.load(f);

index_word_q=dict((word_index_q[i],i) for i in word_index_q);
index_word_a=dict((word_index_a[i],i) for i in word_index_a);
maxlen_q=203;
model=load_model(DIR+"/rn_model.chk");
def infer(data):
    image=Image.open("/home/liangstein/smb1/cnnwork/CLEVR/images/train/"+data[0]);
    image = image.resize((128, 128));
    image_vec=np.array(image)[:,:,:3]*255**-1;
    image_vec=image_vec.reshape((1,128,128,3));
    question_vec=np.zeros((1,maxlen_q),dtype=np.float32);
    for i,ele in enumerate(text_to_word_sequence(data[1])):
        if ele in word_index_q:
            question_vec[0,i]=word_index_q[ele];
    predict=model.predict([question_vec,image_vec])[0];
    answer=index_word_a[np.argmax(predict,axis=0)+1];
    return answer;
