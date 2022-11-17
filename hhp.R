z = read.table("testOutput.txt")

#used to remove the values less than 5% from the column v4 and v5 of our resulting dataset which is tau 1 and tau 2 
tau1 = subset(z , select = -c(V1,V2,V3,V5,V6,V7,V8,V9,V10),z[4]> 5 )
tau2 = subset(z, select = -c(V1,V2,V3,V4,V6,V7,V8,V9,V10),z[5]> 5 )#delets the columns by minus thing...and subset is very useful to do that or to remove the values basically used to filter stuffs
hist(z$V6, breaks = 100 , xlim = c(-190, 190), ylim = c(0,400), xlab = "Dihedral packing angle" , main = "Frequency vs dihedral packing", col = "grey")
zsub = subset(z , select = -c(V1,V2,V3,V6,V7,V10), z$V4>5 & z$V5>5, z$V7<12) # the data which need to be fed in to the machine learning approaches
seq = subset(z , select = -c(V1,V2,V3,V4,V5,V6,V7,V10), z$V4>5 & z$V5>5, z$V7<12 )

#now we have to feed the zsub or our filtered data to the neural network or to the regression approaches
#we should train the data inn 2 cases like firstly with 3 classes and then 6 classes and put the dna sequences as input to cnn and rnn and later the 3 case where i take the whole range of 
#angle from 180 to -180 degrees  will taken as continuous variable and put into regression for training
classA = subset(zsub , select = -c(V4,V5,V6,V7),zsub$V6>= -180 & zsub$V6<= -140)
classB = subset(zsub , select = -c(V4,V5,V6,V7),zsub$V6>= -100 & zsub$V6<= -60 )
classC = subset(zsub , select = -c(V4,V5,V6,V7),zsub$V6>= -50 & zsub$V6<= -10 )
classD = subset(zsub , select = -c(V4,V5,V6,V7),zsub$V6>= 0 & zsub$V6<= 40 )
classE = subset(zsub , select = -c(V4,V5,V6,V7),zsub$V6>= 80 & zsub$V6<= 120 )
classF = subset(zsub , select = -c(V4,V5,V6,V7),zsub$V6>= 130 & zsub$V6<= 170 )
#class1 = append(classA,classD)

write.csv(classE, "classE.csv", quote = FALSE,col.names = FALSE , row.names = FALSE )

library(keras)
rm(list=ls())
# RNN to predict HHP using protein data obtained from PISCES ( culled pdb to give representative dataset)
AA=c("-","A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","
Y") # "-" to allow for zero padding, which is added as a 0
Indx=c(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
Indx_df=data.frame(Indx)
row.names(Indx_df)=AA
maxSeqLength=0
# function to read in seqs and convert to int-values
readSeqsIndx = function(fileName) {
  f= file(fileName, "r")
  seqList=list()
  nSeqs=0
  while (TRUE) {
    line = readLines(f, n = 1)
    if ( length(line) == 0 ) {
      break
    }
    N=nchar(line)
    nSeqs=nSeqs+1
    if(N>maxSeqLength) {maxSeqLength<<-N;}
    Indxvec=c()
    for(aa in rownames(Indx_df)){
      line=gsub(aa,paste(Indx_df[aa,],",",sep=""),line)
    }
    line=substr(line,0,nchar(line)-1)
    Indxvec=as.integer(strsplit(line,",")[[1]])
    seqList[[nSeqs]] =Indxvec
  }
  close(f)
  return(seqList)
}

A=readSeqsIndx("ClassA.txt");
B=readSeqsIndx("classB.txt");
C=readSeqsIndx("ClassC.txt");
D=readSeqsIndx("ClassD.txt");
E=readSeqsIndx("ClassE.txt");
F=readSeqsIndx("ClassF.txt");

#all original sequences of unequal length ??? padding necessary (done below)
length_A =length(A)
length_B =length(B)
length_C =length(C)
length_D =length(D)
length_E =length(E)
length_F =length(F)

yVec=c(rep(0,length_A),rep(1,length_B),rep(2,length_C),rep(3,length_D),rep(4,length_E),rep(5,length_F))
seqs=c(A,B,C,D,E,F)
nTot=length(seqs)
selVec=seq(1,nTot,1)
selVec=sample(selVec) # shuffle data to represent all classes evenly
# 80/20% training/test split
input_train=list()
y_train=c()
input_test=list()
y_test=c()
nTrain=1
nTest=1
for(i in 1:nTot)
{
  if(i< (0.8*nTot)){
    input_train[[nTrain]]=seqs[[selVec[i]]]
    y_train[[nTrain]]=yVec[[selVec[i]]]
    nTrain=nTrain+1
  } else {
    input_test[[nTest]] =seqs[[selVec[i]]]
    y_test[[nTest]] =yVec[[selVec[i]]]
    nTest=nTest+1
  }
}

max_features <- length(AA) # different AAs
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences")
cat("Pad sequences\n") 
input_train <- pad_sequences(input_train, maxlen = maxSeqLength)
input_test <- pad_sequences(input_test, maxlen = maxSeqLength)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

embeddingDim=5 
model <- keras_model_sequential() %>%
layer_embedding(input_dim = max_features, mask_zero=TRUE, output_dim = embeddingDim) %>% # learn AA-embedding into output_dim-dimensional vector; mask_zero = ignore padding 0
layer_simple_rnn(units = 16, return_sequences=FALSE) %>% layer_dense(units = 1, activation = "sigmoid")
summary(model)
  # units=dimensionality of output space
  # return_sequences=TRUE would allow to stack RNN/LSTM layers, default =FALSE
  # return_sequences=TRUE : output for every node
  # return_sequences=FALSE: output for last node only
  # layer_lstm(units=16) %>% # to invoke LSTM architecture, tanh is default activation function
  # returns the last output
model %>% compile(
  optimizer = "adam", #rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)
history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 64,
  validation_split = 0.2
)
plot(history)


