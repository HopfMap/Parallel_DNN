**Parallelization for DNN**

In this repository we study the parallalization strategy for the training of DNN.
In particular we create an ensemble on $n$ DNN wich are trained over $n$ chunks of data. At each epoch the gradient descent is computed and then averaged. The wieghts of each ensemble model are updated accordingly.
The prediction output is compared against that of a single DNN model which is trained over the whole dataset.
The results are interesting, since the MSE (or the accuracy) depends on the number of epochs. Around $n_{epochs}\sim 300$ (for the particular dataset chosen) the enesemble architecture is able to achieve a better accuracy than the single DNN, using $\sim 1/5$ of computing time. 
