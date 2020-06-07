#!/bin/bash
cd ../build

# data set name and path for loading data and saving label
name=delicious
dataPath=../data/$name/
labelPath=../results/label_$name.dat

num_t=1
m=2
nmax=1000
epoch=20
lr=0.1
l1=1
l2=2
muF=true
beta=1
gamma=0.0001 
entropy=true
sparse=true

# training, testing and saving the labels
for ((i=0;i<$num_t;i++));
do
./stream --treeid=$i --seed=$i --name=$name --path=$dataPath --savelabel=$labelPath --loadlabel=$labelPath  \
--mary=$m --nmax=$nmax --epochs=$epoch --lr=$lr --l1=$l1 --l2=$l2 --beta=$beta --gamma=$gamma --muFlag=$muF \
--sparse=$sparse --entropyLoss=$entropy --ens=1
done

# loading the labels
./stream --loadonly=true --treeid=0 --seed=$i --name=$name --path=$dataPath --savelabel=$labelPath --loadlabel=$labelPath \
--mary=$m --nmax=$nmax --epochs=$epoch --lr=$lr --l1=$l1 --l2=$l2 --beta=$beta --gamma=$gamma --muFlag=$muF \
--sparse=$sparse --entropyLoss=$entropy --ens=$num_t

cd -

# # tree parameters
# num_t: default = 1000; number of trees
# m: default = 2; arity of tree; 
# nmax: default = 2; maximum number of nodes in the tree
# epoch: default = 2; number of epochs through the data
# lr: default = 0.1; learning rate

# # objective function parameters
# l1: default = 1; both direction term
# l2: default = 2; purity term 

# # hyper-parameters for re-ranking tail labels
# muF: default = false
# beta: default = 1 = no re-ranking; for amazon data set we use beta=0.8
# gamma: default = 0.0001; another parameter for re-ranking 

# # loss for training regressors
# entropy: default = true; false uses absolute loss

# # sparsity flag; 
# sparse: default = true; only "mediamill" data set is non-sparse thus sparse=false