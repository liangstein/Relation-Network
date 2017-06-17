## Dependency
* Python3.6(numpy, scipy, pickle, h5py),
* Keras2.02,
* Tensorflow v1.1 backend, (Not Theano backend)
* Cuda8.0, Cudnn6.0 (If GPU is used)

## Neural Network Implementation
Latest relation neural network designed by [Deepmind](https://arxiv.org/pdf/1706.01427.pdf). 

## Dataset
The dataset used is [CLEVR: A Diagnostic Dataset for
Compositional Language and Elementary Visual Reasoning](https://cs.stanford.edu/people/jcjohns/clevr/). In fact, only the first 10000 pictures and the corresponding texts are used for training. After 61 epochs, the categorical crossentropy is decreased to 0.0091. The accuracy has reached to 99.88% 

## Effects
Although the training dataset is quite small, the model has shown inferring ability. Larger training dataset can have better inferring effects. The inferring ability of one image that is not in the training dataset is: 

<p align="left">
  <img src="https://github.com/liangstein/Relation-Network/blob/master/CLEVR_train_069999.png" width="300"/>
</p>
```
Question: There is a green thing in front of the sphere to the left of the large cylinder that is behind the purple shiny cylinder; what is it made of?
Predicted answer: rubber
Real answer: rubber

Question: What number of green cylinders have the same material as the large purple cylinder?
Predicted answer: 1
Real answer: 1

Question: What is the shape of the small metal object that is the same color as the small metallic cylinder?
Predicted answer: cube
Real answer: sphere

Question: What is the shape of the purple shiny thing that is the same size as the block?
Predicted answer: cube
Real answer: cylinder

Question: There is a ball that is the same color as the cube; what material is it?
Predicted answer: metal
Real answer: metal

Question: There is a purple cylinder; are there any green metal objects to the left of it?
Predicted answer: yes
Real answer: yes

Question: Are there any big cyan rubber things of the same shape as the small gray rubber object?
Predicted answer: no
Real answer: no

Question: There is a green thing that is right of the purple cylinder; is its shape the same as the thing left of the matte cube?
Predicted answer: yes
Real answer: no

Question: Is there a brown rubber cylinder that has the same size as the metal ball?
Predicted answer: no
Real answer: no
```

## Authors
liangstein (lxxhlb@gmail.com, lxxhlb@mail.ustc.edu.cn)

Thanks to the clear written code from [buriburisuri](https://github.com/buriburisuri/ByteNet), which helps me a lot in understanding the structure of the network. 

