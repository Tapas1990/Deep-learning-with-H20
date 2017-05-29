library(nnet)
library(caret)

library(nnet)
library(caret)

options(digits=3)
set.seed(1234)

#getwd()
digits.data <- read.csv("dataset.csv")

#Structure of the data:
dim(digits.data)
head(colnames(digits.data), 4)
tail(colnames(digits.data), 4)
head(digits.data[1:2, 1:4])

#Let us convert the labels into factor and visualize their
#distribution. We use only the first 5000 images for training and
#the rest for testing purpose 
digits.data$label <- factor(digits.data$label, levels = 0:9)
class(digits.data$label)
i <- 1:5000
digits.X <- digits.data[i, -1]
digits.y <- digits.data[i, 1]




#Now we can train our MLP with the caret wrapper:

digits.m1 <- train(x = digits.X, y = digits.y,
                   method = "nnet",
                   tuneGrid = expand.grid(
                     .size = c(5),
                     .decay = 0.1),
                   trControl = trainControl(method = "none"),
                   MaxNWts = 10000,
                   maxit = 100)


digits.yhat1 <- predict(digits.m1)
caret::confusionMatrix(xtabs(~digits.yhat1 + digits.y))


barplot(table(digits.yhat1))

caret::confusionMatrix(xtabs(~digits.yhat1 + digits.y))

#So, If we want to predict new examples that our model have never seen before:
  

predict(digits.m1, newdata = digits.data[5001:5005,])
digits.y[1:5]

predict(digits.m1, newdata = digits.data[5001:5005,], type = "prob")

Let us visualize the images to understand where the model is failing:
vec <- digits.X[4,]
vec <- data.matrix(vec)
img <- matrix(vec, ncol = 28, byrow = FALSE)
image(img, axes = FALSE, col = grey(seq(1, 0, length = 256)))



#H20 

library(h2o)
h2o.init()
h2o.shutdown()
h2o.init(nthreads = -1)


library(h2o)
localH2O = h2o.init()
digits.data <- read.csv("dataset.csv")
train <- digits.data[1:5000,]
valid <- digits.data[5001:10000,]
test <- digits.data[10001:15000,]

train$label <- as.factor(train$label)
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

model <- h2o.deeplearning(x = 2:785,
                          y = 1,
                          training_frame = train_h2o,
                          hidden = c(10, 10),   #2 hidden layers of 2 units
                          seed=0)

#Predict on test data:
yhat <- h2o.predict(model, test_h2o)
h2o.confusionMatrix(model, test_h2o)


#Predicting Breast cancer:
library(mlbench)
data(BreastCancer)
BreastCancer

dat <- BreastCancer[, -1]  # remove the ID column
dat[, c(1:ncol(dat))] <- sapply(dat[, c(1:ncol(dat))], as.numeric
dat[, 'Class'] <- as.factor(dat[, 'Class'])

train_h2o <- as.h2o(dat[1:300,])
val_h2o <- as.h2o(dat[301:500,])
test_h2o <- as.h2o(dat[501:699,])

model <- 
  h2o.deeplearning(x = 1:9,  # column numbers for predictors
                   y = 10,   # column number for label
                   # data in H2O format
                   training_frame = train_h2o,
                   # or 'Tanh'
                   activation = "TanhWithDropout", 
                   # % of inputs dropout
                   input_dropout_ratio = 0.2, 
                   balance_classes = TRUE,
                   # % for nodes dropout
                   hidden_dropout_ratios = c(0.5, 0.5), 
                   # three layers of 5 units
                   hidden = c(10, 10), 
                   # max. no. of epochs
                   epochs = 10,
                   seed=0) 


h2o.confusionMatrix(model)
#test_h2o
h2o.confusionMatrix(model, val_h2o)






