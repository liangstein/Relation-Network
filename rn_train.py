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
# prepare the dataset
'''DIR="/home/liangstein/smb1/cnnwork/CLEVR/questions";
with open(DIR+'/CLEVR_train_questions.json') as f:
    data = json.load(f);

image_filenames=[];questions=[];answers=[];maxlen_q=0;
for i in np.arange(0,100000):
    ele=data["questions"][i];
    image_filenames.append(ele["image_filename"]);
    questions.append(ele["question"]);
    answers.append(ele["answer"]);
    if maxlen_q<=len(ele["question"]):
        maxlen_q=len(ele["question"]);

all_answer_words=[];all_question_words=[];
for i in np.arange(0,len(questions)):
    q=text_to_word_sequence(questions[i]);
    a=text_to_word_sequence(answers[i]);
    for j in q:all_question_words.append(j);
    for j in a:all_answer_words.append(j);

tokq=Tokenizer();tokq.fit_on_texts(all_question_words);
word_index_q=tokq.word_index;
toka=Tokenizer();toka.fit_on_texts(all_answer_words);
word_index_a=toka.word_index;
DIR=os.getcwd();maxlen_q=203;
with open(DIR+"/image_filenames","wb") as f:
    pickle.dump(image_filenames,f,protocol=pickle.HIGHEST_PROTOCOL);

with open(DIR+"/questions","wb") as f:
    pickle.dump(questions,f,protocol=pickle.HIGHEST_PROTOCOL);

with open(DIR+"/answers","wb") as f:
    pickle.dump(answers,f,protocol=pickle.HIGHEST_PROTOCOL);

with open(DIR+"/word_index_q","wb") as f:
    pickle.dump(word_index_q,f,protocol=pickle.HIGHEST_PROTOCOL);

with open(DIR+"/word_index_a","wb") as f:
    pickle.dump(word_index_a,f,protocol=pickle.HIGHEST_PROTOCOL);

all_image_matrix=[];image_question_index=[[] for _ in np.arange(0,10001)];image_label=0;
DIR="/home/liangstein/smb1/cnnwork/CLEVR/images/train";
for i in np.arange(0,100000-1):
    filename=image_filenames[i];
    if image_filenames[i]==image_filenames[i+1]:
        image_question_index[image_label].append(i);
    elif image_filenames[i]!=image_filenames[i+1]:
        image_question_index[image_label].append(i);
        image_label+=1;
        image=Image.open(DIR+"/"+filename);
        image=image.resize((128,128));
        all_image_matrix.append(np.array(image));
    if i%100==0:print("{}".format(str(i*100000**-1)));

#the last picture
image_question_index[image_label].append(99999);
image=Image.open(DIR+"/"+image_filenames[-1]);image.resize((128,128))
all_image_matrix.append(np.array(image));
all_image_matrix1=[];
for i in all_image_matrix:
    all_image_matrix1.append(i[:,:,:3]); # color picture only needs three dimensions

with open(os.getcwd()+"/all_image_matrix","wb") as f:
    pickle.dump(all_image_matrix1,f,protocol=pickle.HIGHEST_PROTOCOL);

with open(os.getcwd()+"/image_question_index","wb") as f:
    pickle.dump(image_question_index,f,protocol=pickle.HIGHEST_PROTOCOL);'''

# Right now all the dataset needed are saved on files
# Next read the dataset from the files
DIR=os.getcwd();maxlen_q=203;
with open(DIR+"/image_filenames","rb") as f:
    image_filenames=pickle.load(f);

with open(DIR+"/questions","rb") as f:
    questions=pickle.load(f);

with open(DIR+"/answers","rb") as f:
    answers=pickle.load(f);

with open(DIR+"/word_index_q","rb") as f:
    word_index_q=pickle.load(f);

with open(DIR+"/word_index_a","rb") as f:
    word_index_a=pickle.load(f);

index_word_q=dict((word_index_q[i],i) for i in word_index_q);
index_word_a=dict((word_index_a[i],i) for i in word_index_a);
with open(DIR+"/all_image_matrix","rb") as f:
    all_image_matrix=pickle.load(f);

with open(DIR+"/image_question_index","rb") as f:
    image_question_index=pickle.load(f);

batched_image_numbers=100;
del all_image_matrix[8231];del all_image_matrix[-1];#delete not-is-10 batches
while 1:
    if len(all_image_matrix)%batched_image_numbers!=0:
        del all_image_matrix[-1];
    else:
        break;

def generate_batch_data(batched_image_numbers=batched_image_numbers):
    while 1:
        all_image_labels=np.arange(0,len(all_image_matrix));np.random.shuffle(all_image_labels);
        batched_image_labels=np.array_split(all_image_labels,int(len(all_image_labels)*batched_image_numbers**-1));
        for batch in batched_image_labels:
            batch_size=10*batched_image_numbers;
            image_vec=np.zeros((batch_size,128,128,3),dtype=np.float32);
            question_vec=np.zeros((batch_size,maxlen_q),dtype=np.uint8);
            answer_vec=np.zeros((batch_size,len(word_index_a)),dtype=np.bool);
            count=0;
            for i in np.arange(0,len(batch)):
                text_labels=image_question_index[batch[i]];
                for c1,e1 in enumerate(text_labels):
                    for d1,f1 in enumerate(text_to_word_sequence(questions[e1])):
                        question_vec[count+c1,d1]=word_index_q[f1];
                    for j1 in text_to_word_sequence(answers[e1]):
                        answer_vec[count+c1,word_index_a[j1]-1]=1;
                for m1 in np.arange(0,10):
                    image_vec[m1+count]=all_image_matrix[batch[i]]*255**-1;
                count+=10;
            yield ([question_vec,image_vec],answer_vec);

question_input=Input(shape=(maxlen_q,));
question_input1=Embedding(input_length=maxlen_q,input_dim=len(word_index_q)+1,output_dim=64,mask_zero=True)(question_input);
question_output=LSTM(128,return_sequences=False,activation="sigmoid",implementation=2)(question_input1);

image_input=Input(shape=(128,128,3));
image_input1=Conv2D(filters=24,kernel_size=(3,3),strides=(2,2))(image_input);
image_input1=BatchNormalization()(image_input1);
image_input1=Activation("relu")(image_input1);
image_input1=Conv2D(filters=24,kernel_size=(3,3),strides=(2,2))(image_input1);
image_input1=BatchNormalization()(image_input1);
image_input1=Activation("relu")(image_input1);
image_input1=Conv2D(filters=24,kernel_size=(3,3),strides=(2,2))(image_input1);
image_input1=BatchNormalization()(image_input1);
image_input1=Activation("relu")(image_input1);
image_input1=Conv2D(filters=24,kernel_size=(3,3),strides=(2,2))(image_input1);
image_input1=BatchNormalization()(image_input1);
image_input1=Activation("relu")(image_input1);
image_input1=Conv2D(filters=24,kernel_size=(3,3),strides=(2,2))(image_input1);
image_input1=BatchNormalization()(image_input1);
image_output=Activation("relu")(image_input1);

shapes=image_output.shape;w,h=shapes[1],shapes[2];
def slice_width0(t):
    return t[:,0,:,:];

def slice_width1(t):
    return t[:,1:,:,:];

def slice_height0(t):
    return t[:,0,:];

def slice_height1(t):
    return t[:,1:,:];

slice_layer1=Lambda(slice_width0);
slice_layer2=Lambda(slice_width1);
slice_layer3=Lambda(slice_height0);
slice_layer4=Lambda(slice_height1);

features=[];
for k1 in range(w):
    feature_w=slice_layer1(image_output);
    image_output=slice_layer2(image_output);
    for k2 in range(h):
        feature_h=slice_layer3(feature_w);
        feature_w=slice_layer4(feature_w);
        features.append(feature_h);

def G_network(x):
    y = Dense(128)(x);
    y = BatchNormalization()(y);
    y = Activation("relu")(y);
    y = Dense(128)(y);
    y = BatchNormalization()(y);
    y = Activation("relu")(y);
    y = Dense(128)(y);
    y = BatchNormalization()(y);
    y = Activation("relu")(y);
    return y;

def F_network(x):
    y = Dense(128)(x);
    y = BatchNormalization()(y);
    y = Activation("relu")(y);
    y = Dropout(0.1)(y);
    y = Dense(len(word_index_a))(y);
    y = Activation("softmax")(y);
    return y;

G_outputs=[];relations=[];
for feature1 in features:
    for feature2 in features:
        relation=Concatenate()([feature1,feature2,question_output]);
        G_outputs.append(G_network(relation));

F_input=Add()(G_outputs);
output=F_network(F_input);
model=Model(inputs=[question_input,image_input],outputs=output);
opt=adam(lr=0.0003);
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=['accuracy']);
#plot_model(model, to_file="model.png", show_shapes=True);
early = EarlyStopping(monitor="loss", mode="min", patience=10);
lr_change = ReduceLROnPlateau(monitor="loss", factor=0.2, patience=0, min_lr=0.000)
checkpoint = ModelCheckpoint(filepath=DIR + "/rn_model.chk",
                              save_best_only=False);
model.fit_generator(generate_batch_data(),steps_per_epoch=int(len(all_image_matrix) * batched_image_numbers ** -1),
                    nb_epoch=10000, workers=1, callbacks=[early, checkpoint, lr_change], initial_epoch=0);