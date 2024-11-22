
library(MASS)
library(caret)
#library(rda) #Guo et al., 2013
#library(discrim) # fisher and Guo 
#library(rlda)
library(dplyr)
library(mvtnorm)
library(ggplot2)
library(pracma)
library(HiDimDA)
library(MLmetrics)
library(Rdimtools)
library(Matrix)

library(doParallel)
library(foreach)
# Set up a parallel backend with the number of available cores
no_cores <- detectCores() - 1
registerDoParallel(cores=no_cores)
is_positive_semi_definite <- function(mat) {
  if (!is.matrix(mat)) {
    stop("Input must be a matrix")
  }
  
  # Calculate eigenvalues
  eigenvalues <- eigen(mat)$values
  
  # Check if all eigenvalues are greater than or equal to zero
  return(all(eigenvalues >= 0))
}


MPI <- function(matrix) {
  SVD <- svd(matrix);
  DDD <- rep(0, length(SVD$d));
  for (i in 1:length(DDD)) { DDD[i] <- ifelse(SVD$d[i] == 0, 0, 1/SVD$d[i]);  }
  SVD$v %*% diag(DDD) %*% t(SVD$u); }



lda_ginv<- function(x, y) {
  # x is the predictor matrix, y is the response vector -1, 1
  # Compute the means of the predictors for each class
  n <- length(y)
  p <- ncol(x)
  # classes <- unique(y)
  # n_classes <- length(classes)
  #mu <- matrix(0, nrow = p, ncol = n_classes)
  mu <- matrix(0, nrow = p, ncol =2)
  S <- matrix(0, nrow = p, ncol = p)
  #for (i in 1:n_classes) {
  #  mu[, i] <- colMeans(x[y == classes[i], ])} 
  if ( p == 1) {
    mu[, 1] <- mean(x[y == 1, ])
    mu[, 2] <- mean(x[y == -1, ])
    S <- cov(x)
  }
  else {
    mu[, 1] <- colMeans(x[y == 1, ])
    mu[, 2] <- colMeans(x[y == -1, ])
    S <- (cov(x[y == 1, ])  + cov(x[y == -1, ]) )/2
  }

  S_inv <- ginv(S)
  #S_inv <- MPI(S)
  # Compute the coefficients of the linear discriminants
  dif_vec <- t(t(as.vector(mu[, 1] - mu[, 2])))
  sum_vec <- t(t(as.vector(mu[, 1] + mu[, 2])))
  coef <- S_inv %*% dif_vec  #beta1 coefficient of x
  # Compute the threshold for classification
  thresh <- (1/2) * t(coef) %*% sum_vec  #beta0, intercept, used to compare 
  return(list(coef = coef, thresh = thresh, prec_matrix = S_inv, group_mean = mu))
}; predict_lda <- function(newdata, model) {
  score <-  as.vector(t(model$coef) %*% t(newdata))  
    pred <- ifelse(score > as.numeric(model$thresh), 1, -1)
  return(pred)
}

n = 50
result_sim1 <- NULL;repeated_test_error <- NULL; p_result <- NULL; 
set.seed(123)
foreach(i = 1:200, .packages = c("caret","MASS")) %dopar% {
#  for (i in 1:200){
  p = i
  for ( ind in 1: 50 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(1.3,p);  sigma <- diag(p)
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = 0.7, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:(ncol(train)-1)]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:(ncol(test)-1)]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    # train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    #train.error_ori <- mean(pred!=test.y)
    #f1_score <- c(f1_score, F1_Score(test$response, pred2))
    # repeated_train_error <- c(repeated_train_error,train_error)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  #result_sim1 <- c(result_sim1, mean(repeated_test_error))
  result_sim1 <- mean(repeated_test_error)
 # result_se <- sd(repeated_test_error)
 # p_result <- c( p_result, p)
  p_result <- p
  return(c(result_sim1 = result_sim1, p = p_result))
}-> results

n100 <- cbind(1:200, result_sim1) 
n60 <- cbind(1:200, result_sim1)
n140 <- cbind(1:200, result_sim1)
plot(1:200, result_sim1)
abline(v = 70)

result <- do.call(rbind, results)
plot(result[,2], result[,1])



delta = 0.5
delta = 1
delta = 1.5

nk_seq <- floor(seq(50, 100, length=20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.01, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7

result_sim1 <- NULL;repeated_test_error <- NULL; gamma_result <- NULL; 
set.seed(123)
foreach(i = 1:20, .packages = c("caret","MASS")) %dopar% {
    p = p_seq[i]
    n = nk_seq[i]
  for ( ind in 1: 80 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p);  sigma <- diag(p)
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
   # train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    #train.error_ori <- mean(pred!=test.y)
    #f1_score <- c(f1_score, F1_Score(test$response, pred2))
   # repeated_train_error <- c(repeated_train_error,train_error)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train)) 
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
}-> results

plot(results[,3],result[,1])
result_sim1
gamma_result
del05 = cbind(result_sim1, gamma_result)
del05 = rbind(c(0,0.3608333,1),del05)
del1  = cbind(result_sim1, gamma_result)
del1 = rbind(c(0,0.2895833,1),del1)  #0.21667
del15 = cbind(result_sim1, gamma_result)
del15 = rbind(c(0,0.114,1),del15)


plot(gamma_result, result_sim1[,2])
plot(gamma_result, result_sim1[,1])
df <- cbind(del05, del1, del15)
setwd("C:/Users/kehan/OneDrive/Desktop/research/code")
write.csv(df, "identity_empirical_gamma0to5.csv", row.names = FALSE)


#gamma  = 1 
repeated_train_error <- NULL
repeated_test_error <- NULL
delta = 0.5
delta = 1
delta = 1.5
for ( ind in 1: 80 ) {
  mu1 <- rep(0,50) ; mu2 <- rep(0+delta,50);  sigma <- diag(50)
  group1 <- mvrnorm(35,mu1,sigma); group2 <- mvrnorm(35,mu2,sigma)
  response_g1 <- rep(1,35); response_g2 <- rep(-1,35); response <- c(response_g1,response_g2)
  
  data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
  #data$response <- as.factor(data$response)
  trainIndex <- createDataPartition(data$response, p = 0.7, list = FALSE)
  
  train <- data[trainIndex, ];  test <- data[-trainIndex, ]
  train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
  test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
  mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
  train_error <- mean(as.numeric(as.character(pred))!=train.y)
  #pred2 <- predict(mod2, test)$class
  pred2 <- predict_lda(test.x,mod)
  test_error <- mean(as.numeric(as.character(pred2))!=test.y)
  #train.error_ori <- mean(pred!=test.y)
  #f1_score <- c(f1_score, F1_Score(test$response, pred2))
  repeated_train_error <- c(repeated_train_error,train_error)
  repeated_test_error <- c(repeated_test_error,test_error)
  gamma_result2 <- ncol(train)/(nrow(train))
}
mean(repeated_test_error) #0.3608333   0.2895833   0.215

plot(del05[2:21,3],del05[2:21,2], col= "black", type = "l" , pch=20,
     ylim=c(0,0.4),xlim =c(0,4),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "Identity Covariance Matrix")
points(del1[2:21,3],del1[2:21,2],col="darkred", lwd=1.2,type = "l", pch = 20)
points(del15[2:21,3],del15[2:21,2],col="darkblue",lwd=1.2,type = "l" ,pch=20)
#abline( = 1, col = "black")
points(del05[2:21,3],del05[2:21,2], pch=3, col = "black")
points(del1[2:21,3],del1[2:21,2],pch=8,col="darkred")
points(del15[2:21,3],del15[2:21,2], pch=6,col="darkblue")
legend(x = 2.5, y = 0.4, cex =1.2, legend = expression(paste(delta, " = 0.5"), paste(delta, " = 1"), paste(delta, " = 1.5")), 
       col = c("black", "darkred", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(3, 8, 6),
       bty = "n")

#############################################################################



#######################Compound Symmetric###########################################
#sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- 0.3


delta = 1
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.01, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

rho = 0.3
rho = 0.5 
rho = 0.1
partition = 0.7
#result_sim1 <- NULL; f1_score <- NULL; repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL; train_row <- NULL
set.seed(123)
result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL; results  <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar% {
  p = p_seq[i]
  n = nk_seq[i]
  repeated_test_error <- NULL 
  for ( ind in 1: 80 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(1,p); sigma <- rWishart(1,n*2,diag(p))[, , 1]; sigma <- sigma/(n*2 -1) #sigma <- diag(p); sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- 2 ; sigma <- nearPD(sigma)$mat
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    lda_mod <- lda(response ~., data = train)
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    #train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    #train.error_ori <- mean(pred!=test.y)
    #f1_score <- c(f1_score, F1_Score(test$response, pred2))
   # repeated_train_error <- c(repeated_train_error,train_error)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train))  # Make sure 'train' is defined correctly here
  return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
} -> results

plot(results[,3], results[,1], type = "b")
abline(v = 1)
rho2 <- results
wishart <- results

plot(wishart[,3], wishart[,1], type = b, xlab= expression(gamma), ylab = "Error Rate", Main = "Wishart Covariance Matrix")


df2_ro03 <- cbind(result_sim1, gamma_result)
df2_ro05 <- cbind(result_sim1, gamma_result)
df2_ro01 <- cbind(result_sim1, gamma_result)
df2 <- cbind(df2_ro01, df2_ro03, df2_ro05)
write.csv(df2, "compoundsymmetric_empirical_new.csv", row.names = FALSE)
df_cs <- read.csv("compoundsymmetric_empirical_new.csv", header = T)
plot(gamma_result, result_sim1[,2])

plot(df_cs[,3],df_cs[,2], col= "#c44601", type = "l" , ylim=c(0,0.55),xlim =c(0,4),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "Compound Symmetric Covariance Matrix")
lines(df_cs[,3],df_cs[,5],col="#5ba300",lwd=1.2)
lines(df_cs[,3],df_cs[,8],col="#054fb9",lwd=1.2)
#abline( = 1, col = "black")
points(df_cs[,3],df_cs[,2], pch=17, col = "#c44601")
points(df_cs[,3],df_cs[,5],pch=15,col="#5ba300")
points(df_cs[,3],df_cs[,8], pch=19,col="#054fb9")
legend(x = 2, y = 0.5, cex =1.2, legend = expression(paste(cov(x[i], x[j]), " = 0.1"), paste(cov(x[i], x[j]), " = 0.3"), paste(cov(x[i], x[j]), " = 0.5")), 
       col = c("#c44601", "#5ba300", "#054fb9"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(17, 15, 19),
       bty = "n")

#Wishart
wishart1 <- results
write.csv(wishart1, "wishart1.csv", row.names = FALSE)

sdelta = 1
nk_seq <- floor(seq(20, 50, length=100)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq <- seq(0.01, 4, length=100) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7
result_sim2 <- NULL;  repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result2 <- NULL; train_row <- NULL
set.seed(123)
for (ij in 1:45) {
  p = p_seq[ij]
  n = n_full[ij]
  for ( ind in 1:20) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p);  sigma <- diag(p); sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- 2
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    #train.error_ori <- mean(pred!=test.y)
    #f1_score <- c(f1_score, F1_Score(test$response, pred2))
    repeated_train_error <- c(repeated_train_error,train_error)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim2 <- rbind(result_sim2,c(mean(repeated_train_error),mean(repeated_test_error)))
  gamma_result2 <- c(gamma_result2, p/(nrow(train)))
  train_row <- c(train_row, nrow(train))
  repeated_train_error <- NULL
  repeated_test_error <- NULL
}
result_sim2
gamma_result2
df2_ro01 <- cbind(result_sim2, gamma_result2)
df2_ro03 <- cbind(result_sim2, gamma_result2)
df2_ro05 <- cbind(result_sim2, gamma_result2)
write.csv(df2, "compoundsymmetric_empirical_new", row.names = FALSE)

df2_ro <- cbind(gamma_result2, df2_ro01[,2], df2_ro03[,2], df2_ro05[,2] )
write.csv(df2_ro, "compoundsymmetric_empirical_gamma0to1_3ro_delta1", row.names = FALSE)

plot(gamma_result2, result_sim2[,2])
plot(gamma_result2, result_sim2[,1])

df2 <- data.frame(Gamma=gamma_result2, ro01=df2_ro01[,2], ro03 = df2_ro03[,2], ro05 = df2_ro05[,2])


#################AR1 covariance matrix ####################################################################

gamma_seq1 <- seq(0.01, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

rho = 0.3
rho = 0.7 
rho = 0.5

partition = 0.7
result_sim1 <- NULL; f1_score <- NULL; repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL;
set.seed(123)
for (ij in 1:20) {
  p = p_seq[ij]
  n = nk_seq[ij]
  for ( ind in 1: 80 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p); sigma = rho^abs(outer(1:p, 1:p, "-"))
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    #train.error_ori <- mean(pred!=test.y)
    #f1_score <- c(f1_score, F1_Score(test$response, pred2))
    repeated_train_error <- c(repeated_train_error,train_error)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- rbind(result_sim1,c(mean(repeated_train_error),mean(repeated_test_error)))
  gamma_result <- c(gamma_result, p/(nrow(train)))
  train_row <- c(train_row, nrow(train))
  repeated_train_error <- NULL
  repeated_test_error <- NULL
}

delta = 1;partition = 0.7;
result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL ;result_sim2 <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%
  {
    p <- p_seq[i]
    n <- nk_seq[i]
    repeated_test_error <- NULL; repeated_train_error = NULL
    #sigma <- diag(p) 
    sigma = rho^abs(outer(1:p, 1:p, "-"))
    mu1 <- rep(0,p) ; mu2 <- rep(1,p); 
    for (ind in 1:80) {
      #sigma <- diag(p);#sigma <- diag(p); sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- -0.5
      #sigma <- generate_sparse_matrix(p, 1)
      group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
      response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
      data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
      trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
      train <- data[trainIndex, ];  test <- data[-trainIndex, ]
      train.x <- as.matrix(train[,1:(ncol(train)-1)]); train.y <- (train[,ncol(train)])
      test.x <- as.matrix(test[,1:(ncol(test)-1)]); test.y <- (test[,ncol(test)])
      mod <- lda_ginv(train.x,train.y)
      pred2 <- predict_lda(test.x,mod)
      test_error <- mean(as.numeric(as.character(pred2)) != test.y)
      pred <- predict_lda(train.x,mod) 
     # train_error <- mean(as.numeric(as.character(pred))!=train.y)
      repeated_test_error <- c(repeated_test_error,test_error)
      #repeated_train_error <- c(repeated_train_error,train_error)
    }
    result_sim2 <- mean(repeated_train_error)
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train)) 
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result, result_sim2 = result_sim2))
  } -> results

plot(results[,3], results[,1])
df3_ro03 <- cbind(results[,3], results[,1])
df3_ro05 <- cbind(results[,3], results[,1])
df3_ro07 <- cbind(results[,3], results[,1])

df3 <- cbind(df3_ro03, df3_ro05, df3_ro07)
write.csv(df3, "ar1_empirical_new.csv", row.names = FALSE)
plot(gamma_result, result_sim1[,2])
plot(df3_ro03[,1],df3_ro03[,2], col= "#8e0bca", type = "l" , ylim=c(0,0.4),xlim =c(0,3),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "AR(1) Covariance Matrix")
lines(df3_ro05[,1],df3_ro05[,2],col="#e1111e",lwd=1.2)
lines(df3_ro07[,1],df3_ro07[,2],col="#08b128",lwd=1.2)
#abline( = 1, col = "black")
points(df3_ro03[,1],df3_ro03[,2], pch=17, col = "#8e0bca")
points(df3_ro05[,1],df3_ro05[,2],pch=15,col="#e1111e")
points(df3_ro07[,1],df3_ro07[,2], pch=19,col="#08b128")
legend(x = 2, y = 0.4, cex =1, legend = expression(paste(rho, " = 0.3"), paste(rho, " = 0.5"), paste(rho, " = 0.7")), 
       col = c("#8e0bca", "#e1111e", "#08b128"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(17, 15, 19),
       bty = "n")
rho = 0.7  #0.3  0.5 0.7,; Sigma = rho^abs(outer(1:p, 1:p, "-"))
delta = 1
nk_seq <- floor(seq(20, 100, length=100)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq <- seq(0.01, 5, length=100) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7
result_sim3 <- NULL;repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result3 <- NULL; train_row <- NULL
set.seed(123)
for (ij in 1:45) {
  p = p_seq[ij]
  n = n_full[ij]
  for ( ind in 1: 20 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p);  sigma <- diag(p); sigma = rho^abs(outer(1:p, 1:p, "-"))
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    #train.error_ori <- mean(pred!=test.y)
    #f1_score <- c(f1_score, F1_Score(test$response, pred2))
    repeated_train_error <- c(repeated_train_error,train_error)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim3 <- rbind(result_sim3,c(mean(repeated_train_error),mean(repeated_test_error)))
  gamma_result3 <- c(gamma_result3, p/(nrow(train)))
  train_row <- c(train_row, nrow(train))
  repeated_train_error <- NULL
  repeated_test_error <- NULL
}
result_sim3
gamma_result3
df3_ro03 <- cbind(result_sim3, gamma_result3) #43
df3_ro05 <- cbind(result_sim3, gamma_result3)
df3_ro07 <- cbind(result_sim3, gamma_result3)
df3_save <- cbind(gamma_result3[1:43],df3_ro03, df3_ro05[1:43,], df3_ro07[1:43,])
write.csv(df3_save, "ar1covmatrix_empirical_gamma0to1.csv", row.names = FALSE)


plot(gamma_result3, result_sim3[,2])
plot(gamma_result2, result_sim2[,1])

#############RANDOM ENTIRES################################################################
#high sparsity, less 0, more filled 
gamma_seq1 <- seq(0.01, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

generate_cov_matrix <- function(dim) {
  # Step 1: Create a matrix with random off-diagonal entries from -1 to 1
  mat <- matrix(runif(dim*dim, min = -1, max = 0), ncol = dim)
  
  # Step 2: Set the diagonal entries to 1
  diag(mat) <- 1
  
  # Step 3: Make the matrix symmetric
  mat <- (mat + t(mat))/2
  
  # Step 4: Make the matrix positive semi-definite
  #eig <- eigen(mat)
  #eig$values <- pmax(eig$values, 1e-6)
  #mat <- mat +diag(dim)
  #diag(mat) <- alpha * diag(mat) + (1 - alpha)
  mat <- nearPD(mat)$mat
  
  return(mat)
}
generate_sparse_matrix <- function(p, sparsity) {
  A <- matrix(0, nrow=p, ncol=p)
  for (i in 1:p) {
    for (j in 1:p) {
      if (runif(1) < sparsity) { 
        A[i, j] <- runif(1, min=-1, max=1)
      }
    }
  }
  #Make the matrix symmetric
  A <- (A + t(A)) / 2
  # Step 3: Add a multiple of the identity matrix to make it positive definite
  A <- A + diag(p)
  A_nearest_pd <- nearPD(A)$mat
  return(A_nearest_pd)
}

# Fix the covariance matrix 

set.seed(200); SIGMA <- generate_sparse_matrix(300,0.4)#SIGMA <- generate_cov_matrix(300)

is_positive_semi_definite <- function(mat) {
  # Ensure the matrix is symmetric to avoid incorrect results
  if (!all(mat == t(mat))) {
    stop("Matrix is not symmetric")
  }
  
  # Get eigenvalues of the matrix
  eig_values <- eigen(mat)$values
  
  # Check if all eigenvalues are non-negative (tolerating very small negative values due to numerical errors)
  return(all(eig_values > -1e-10))
}

delta = 1;partition = 0.7;
result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL ;result_sim2 <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%
  {
    p <- p_seq[i]
    n <- nk_seq[i]
    # Use foreach to replace the inner loop
    repeated_test_error <- NULL; repeated_train_error = NULL
   # sigma <- diag(p); sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- -0.7; sigma <- nearPD(sigma)$mat
    mu1 <- rep(0,p) ; mu2 <- rep(1,p); 
    for (ind in 1:300) {
      #sigma <- diag(p);#sigma <- diag(p); sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- -0.5
      #sigma <- generate_sparse_matrix(p, 1)
      sigma <- SIGMA[1:p,1:p];
      group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
      response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
      data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
      trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
      train <- data[trainIndex, ];  test <- data[-trainIndex, ]
      train.x <- as.matrix(train[,1:(ncol(train)-1)]); train.y <- (train[,ncol(train)])
      test.x <- as.matrix(test[,1:(ncol(test)-1)]); test.y <- (test[,ncol(test)])
      mod <- lda_ginv(train.x,train.y)
      pred2 <- predict_lda(test.x,mod)
      test_error <- mean(as.numeric(as.character(pred2)) != test.y)
      pred <- predict_lda(train.x,mod) 
      train_error <- mean(as.numeric(as.character(pred))!=train.y)
      # Return the test_error for each iteration
      repeated_test_error <- c(repeated_test_error,test_error)
      repeated_train_error <- c(repeated_train_error,train_error)
    }
    result_sim2 <- mean(repeated_train_error)
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train))  # Make sure 'train' is defined correctly here
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result, result_sim2 = result_sim2))
  } -> results


results_01 <- results
results_02 <- results
results_051 <- results
results_neg11 <- results
results_neg10 <- results
results_neg20 <- results
results_neg30 <- results 
results_neg0505 <- results
results_neg0303 <- results
results_neg22 <- results 

sparse_02 <- results


plot(results[,3], results[,1], type = "b")
plot(results[,3], results[,4], type = "b")
abline(v = 1)
abline(v = 1.7)
plot(results[,3], results[,2], type = "b")
results_01 <- results
results_051 <- results
results_neg11 <- results
results_neg10 <- results
results_neg0505 <- results
results_neg0303 <- results
df_random <- cbind(results_neg0505, results_neg11, results_01)
write.csv(df_random, "random_unif_empirical_negative.csv", row.names = FALSE)
write.csv(results_02, "random_unif_empirical_02.csv", row.names = FALSE)

write.csv(results_01, "random_unif_empirical_01NEW.csv", row.names = FALSE)
write.csv(results_051, "random_unif_empirical_051NEW.csv", row.names = FALSE)
write.csv(results_neg11, "random_unif_empirical_neg11NEW.csv", row.names = FALSE)
write.csv(results_neg10, "random_unif_empirical_neg10NEW.csv", row.names = FALSE)
write.csv(results_neg20, "random_unif_empirical_neg20NEW.csv", row.names = FALSE)
write.csv(results_neg30, "random_unif_empirical_neg30NEW.csv", row.names = FALSE)
write.csv(results_neg0505, "random_unif_empirical_neg0505NEW.csv", row.names = FALSE)
write.csv(results_neg0303, "random_unif_empirical_neg0303NEW.csv", row.names = FALSE)




results_neg050 <- results
results_neg01 <- results
results_neg03 <- results
results_neg07 <- results

df_neg_compound <- cbind(results_neg01, results_neg050, results_neg07)
write.csv(df_neg_compound, "compoundsymmetric_empirical_negative.csv", row.names = FALSE)

plot(results_neg0505[1:20,3], results_neg0505[1:20,1], ylim = c(0,0.55), main = "Random Entries Covariance Matrix (Empirical)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_neg0505[1:20,3], results_neg0505[1:20,1], lwd = 1.2, col = "darkred")
points(results_neg11[1:20,3], results_neg11[1:20,1], pch = 25, col = "darkblue")
lines(results_neg11[1:20,3], results_neg11[1:20,1], lwd =1.2, col = "darkblue")
points(results_01[1:20,3], results_01[1:20,1], pch = 15, col = "orange")
lines(results_01[1:20,3], results_01[1:20,1], lwd =1.2, col = "orange")
points(results_02[1:20,3], results_02[1:20,1], pch = 15, col = "pink")
lines(results_02[1:20,3], results_02[1:20,1], lwd =1.2, col = "pink")
points(results_neg0303[1:20,3], results_neg0303[1:20,1], pch = 17, col = "darkgreen")
lines(results_neg0303[1:20,3], results_neg0303[1:20,1], lwd =1.2, col = "darkgreen")
legend(x =2.5, y = 0.55,
       legend = expression(paste("unif(-0.5,0.5)"), paste("unif(-1,1)"), paste("unif(-0.3,-0.3)"), paste("unif(0,1)")), 
       col = c("darkred","darkblue","darkgreen", "orange"), bty = "n",
       lty = 1, 
       pch = c(19,25,17, 15),
       cex = 1, 
       text.col = "black")

plot(results_neg10[1:20,3], results_neg10[1:20,1], ylim = c(0,0.55), main = "Random Entries Covariance Matrix (Empirical)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_neg10[1:20,3], results_neg10[1:20,1], lwd = 1.2, col = "darkred")
points(results_01[1:20,3], results_01[1:20,1], pch = 25, col = "darkblue")
lines(results_01[1:20,3], results_01[1:20,1], lwd =1.2, col = "darkblue")
#points(results_neg30[1:20,3], results_neg30[1:20,1], pch = 15, col = "orange")
#lines(results_neg30[1:20,3], results_neg30[1:20,1], lwd =1.2, col = "orange")
#points(results_02[1:20,3], results_02[1:20,1], pch = 15, col = "pink")
#lines(results_02[1:20,3], results_02[1:20,1], lwd =1.2, col = "pink")
points(results_neg11[1:20,3], results_neg11[1:20,1], pch = 17, col = "darkgreen")
lines(results_neg11[1:20,3], results_neg11[1:20,1], lwd =1.2, col = "darkgreen")
legend(x =2.5, y = 0.55,
       legend = expression(paste("unif(-1,0)"), paste("unif(0,1)"), paste("unif(-1,1)")), 
       col = c("darkred","darkblue","darkgreen"), bty = "n",
       lty = 1, 
       pch = c(19,25,17),
       cex = 1, 
       text.col = "black")


plot(results_neg20[1:20,3], results_neg20[1:20,1], ylim = c(0,0.55), main = "Random Entries Covariance Matrix (Empirical)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_neg20[1:20,3], results_neg20[1:20,1], lwd = 1.2, col = "darkred")
points(results_02[1:20,3], results_02[1:20,1], pch = 25, col = "darkblue")
lines(results_02[1:20,3], results_02[1:20,1], lwd =1.2, col = "darkblue")
#points(results_neg30[1:20,3], results_neg30[1:20,1], pch = 15, col = "orange")
#lines(results_neg30[1:20,3], results_neg30[1:20,1], lwd =1.2, col = "orange")
#points(results_02[1:20,3], results_02[1:20,1], pch = 15, col = "pink")
#lines(results_02[1:20,3], results_02[1:20,1], lwd =1.2, col = "pink")
points(results_neg22[1:20,3], results_neg22[1:20,1], pch = 17, col = "darkgreen")
lines(results_neg22[1:20,3], results_neg22[1:20,1], lwd =1.2, col = "darkgreen")
legend(x =2.5, y = 0.55,
       legend = expression(paste("unif(-2,0)"), paste("unif(0,2)"), paste("unif(-2,2)")), 
       col = c("darkred","darkblue","darkgreen"), bty = "n",
       lty = 1, 
       pch = c(19,25,17),
       cex = 1, 
       text.col = "black")



plot(results_neg050[1:20,3], results_neg050[1:20,1], ylim = c(0.3,0.55), main = "Compound Symmetric Covariance Matrix (Empirical)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_neg050[1:20,3], results_neg050[1:20,1], lwd = 1.2, col = "darkred")
#points(results_sparse_50[1:20,3], results_sparse_50[1:20,1], pch = 17, col = "darkgreen")
#lines(results_sparse_50[1:20,3], results_sparse_50[1:20,1], lwd =2, col = "darkgreen")
points(results_neg07[1:20,3], results_neg07[1:20,1], pch = 25, col = "darkblue")
lines(results_neg07[1:20,3], results_neg07[1:20,1], lwd =1.2, col = "darkblue")
points(results_neg03[1:20,3], results_neg03[1:20,1], pch = 15, col = "orange")
lines(results_neg03[1:20,3], results_neg03[1:20,1], lwd = 1.2, col = "orange")
legend(x =3, y = 0.45,
       legend = expression(paste(rho, " = -0.3"), paste(rho, " = -0.5"), paste(rho, " = -0.7")), 
       col = c("orange","darkred","darkblue"), bty = "n",
       lty = 1, 
       pch = c(15,19,25),
       cex = 1, 
       text.col = "black")

write.csv(results_neg050, "cs_negative_empirical_neg050.csv", row.names = FALSE)
write.csv(results_neg07, "cs_negative_empirical_neg07.csv", row.names = FALSE)
write.csv(results_neg03, "cs_negative_empirical_neg03.csv", row.names = FALSE)



#############sparse covariance matrix ##################################################################

generate_sparse_matrix <- function(p, sparsity) {
  A <- matrix(0, nrow=p, ncol=p)
  for (i in 1:p) {
    for (j in 1:p) {
      if (runif(1) < sparsity) { 
        A[i, j] <- runif(1, min=0, max=1)
      }
    }
  }
  #Make the matrix symmetric
  A <- (A + t(A)) / 2
  # Step 3: Add a multiple of the identity matrix to make it positive definite
  A <- A + diag(p)
  A_nearest_pd <- nearPD(A)$mat
  return(as.matrix(A_nearest_pd))
}
#high sparsity, less 0, more filled ]=
generate_cov_matrix <- function(dim) {
  # Step 1: Create a matrix with random off-diagonal entries from -1 to 1
  mat <- matrix(runif(dim*dim, min = 0, max = 1), ncol = dim)
  
  # Step 2: Set the diagonal entries to 1
  diag(mat) <- 1
  
  # Step 3: Make the matrix symmetric
  mat <- (mat + t(mat))/2
  
  # Step 4: Make the matrix positive semi-definite
  #eig <- eigen(mat)
  #eig$values <- pmax(eig$values, 1e-6)
  #mat <- mat +diag(dim)
  #diag(mat) <- alpha * diag(mat) + (1 - alpha)
  mat <- nearPD(mat)$mat
  
  return(mat)
}


delta = 1
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.1, 1, length=10) 
gamma_seq2 <- seq(1.02, 4, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7

set.seed(1)
SIGMA <- generate_sparse_matrix(400,0.9)

set.seed(1);SIGMA <- generate_sparse_matrix(400,0.5)
set.seed(1);SIGMA <- generate_sparse_matrix(400,0.1)

delta = 1
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.1, 1, length=10) 
gamma_seq2 <- seq(1.02, 4, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7
result_sim1 <- NULL;repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL; train_row <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%  {
  #for (i in 1:20) {
  p = p_seq[i]
  n = nk_seq[i]
  repeated_test_error <- NULL
  for ( ind in 1: 80 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p);  sigma <- SIGMA[1:p,1:p]
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train)) 
  return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
}-> results
plot(results[,3], results[,1], type = "b")

sparse_90fill <- results
sparse_50fill <- results
sparse_10fill <- results

plot(sparse_90fill[1:20,3], sparse_90fill[1:20,1], ylim = c(0,0.5), main = "Sparse Covariance Matrix", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(sparse_90fill[1:20,3], sparse_90fill[1:20,1], lwd = 1.2, col = "darkred")
points(sparse_50fill[1:20,3], sparse_50fill[1:20,1], pch = 17, col = "darkgreen")
lines(sparse_50fill[1:20,3], sparse_50fill[1:20,1], lwd =1.2, col = "darkgreen")
points(sparse_10fill [1:20,3], sparse_10fill[1:20,1], pch = 25, col = "darkblue")
lines(sparse_10fill[1:20,3], sparse_10fill[1:20,1], lwd =1.2, col = "darkblue")
legend(x =3, y = 0.45, title = "Sparsity Level",
       legend = c("Level = 1",
                  "Level = 5",
                  "Level = 9"),
       col = c("darkred","darkgreen","darkblue"), bty = "n",
       lty = 1, 
       pch = c(19,17,25),
       cex = 1, 
       text.col = "black")


write.csv(sparse_90fill, "empirical_sparse_90.csv", row.names = FALSE)
write.csv(sparse_50fill, "empirical_sparse_50.csv", row.names = FALSE) 
write.csv(sparse_10fill , "empirical_sparse_10.csv", row.names = FALSE)




 #result_sim1 <- NULL;repeated_train_error <- NULL;; gamma_result <- NULL; train_row <- NULL
result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL;repeated_test_error <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%
  {
    p <- p_seq[i]
    n <- nk_seq[i]
    # Use foreach to replace the inner loop
    repeated_test_error <- NULL
    for (ind in 1:100) {
      mu1 <- rep(0,p) ; mu2 <- rep(1,p); 
      sigma <- SIGMA[1:p,1:p];
      #sigma <- generate_cov_matrix(p)
      #sigma <- generate_sparse_matrix(p, 1)
      group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
      response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
      data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
      trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
      train <- data[trainIndex, ];  test <- data[-trainIndex, ]
      train.x <- as.matrix(train[,1:(ncol(train)-1)]); train.y <- (train[,ncol(train)])
      test.x <- as.matrix(test[,1:(ncol(test)-1)]); test.y <- (test[,ncol(test)])
      mod <- lda_ginv(train.x,train.y)
      pred2 <- predict_lda(test.x,mod)
      test_error <- mean(as.numeric(as.character(pred2)) != test.y)
      # Return the test_error for each iteration
      repeated_test_error <- c(repeated_test_error,test_error)
    }
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train))  # Make sure 'train' is defined correctly here
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
  } -> results

results_sparse_20 <- read.csv("empirical_sparse_20.csv")
results_sparse_70 <- read.csv("empirical_sparse_70.csv")
results_sparse_05 <- read.csv("empirical_sparse_05.csv")
results_sparse_50 <- read.csv("empirical_sparse_50.csv")
plot(results_sparse_20[1:20,3], results_sparse_20[1:20,1], ylim = c(0,0.5), main = "Sparse Covariance Matrix", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_sparse_20[1:20,3], results_sparse_20[1:20,1], lwd = 1.2, col = "darkred")
points(results_sparse_50[1:20,3], results_sparse_50[1:20,1], pch = 17, col = "darkgreen")
lines(results_sparse_50[1:20,3], results_sparse_50[1:20,1], lwd =1.2, col = "darkgreen")
points(results_sparse_70[1:20,3], results_sparse_70[1:20,1], pch = 25, col = "darkblue")
lines(results_sparse_70[1:20,3], results_sparse_70[1:20,1], lwd =1.2, col = "darkblue")
#points(results_sparse_05[1:20,3], results_sparse_05[1:20,1], pch = 15, col = "orange")
#lines(results_sparse_05[1:20,3], results_sparse_05[1:20,1], lwd =1.2, col = "orange")

legend(x =3, y = 0.45, title = "Sparsity Level",
       legend = c("Level = 80",
                  "Level = 50",
                  "Level = 30"),
       col = c("darkred","darkgreen","darkblue"), bty = "n",
       lty = 1, 
       pch = c(19,17,25),
       cex = 1, 
       text.col = "black")
plot(gamma_result, result_se, type = "b")

results_sparse_05 <- results
results_sparse_10 <- results
results_sparse_20 <- results
results_sparse_50 <- results
results_sparse_70 <- results
results_sparse_1 <- results
write.csv(results_sparse_05, "empirical_sparse_05.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_sparse_20, "empirical_sparse_20.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_sparse_30, "empirical_sparse_30.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_sparse_50, "empirical_sparse_50.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_sparse_70, "empirical_sparse_70.csv", row.names = FALSE) #variance 3, mean 1,0 
plot(results_tdist[,3], results_tdist[,1], type = "b")

plot(results[,3], results[,1], type = "b")
abline(v= 2)
abline(v = 1)
plot(results[,3], results[,2], type = "b")




#########Mis-specified Model#########################################################

# p = 100,

#generate data from logistic regresssion 

#true model 
p = 500
#Identity Matrix
sigma = diag(p)
diag(sigma) = 5
#mu1 <- runif(p, 1,1) 
#mu2 <- runif(p,0,1)
mu1 <- rep(1,p) 
mu2 <- rep(0,p)
group1 <- mvrnorm(50,mu1,sigma);
group2 <- mvrnorm(50,mu2,sigma)
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.105556, 4.2, length=10) 
gamma_seq <- c(gamma_seq1,gamma_seq2)
response_g1 <- rep(1,50); response_g2 <- rep(-1,50); response <- c(response_g1,response_g2)
data <- rbind(group1,group2); 
data <- as.data.frame(cbind(data,response))


# Generate from logistic 
n <- 100
p <- 300 
X <- matrix(rnorm(n * p), nrow = n, ncol = p)
beta <- rnorm(p)
linear_comb <- X %*% beta
prob_Y1 <- 1/(1 + exp(-linear_comb))
Y <- rbinom(n, size = 1, prob_Y1)
Y[Y == 0] <- -1
data <- data.frame(X, Y)

library(Matrix)
set.seed(123) # For reproducibility
#p <- 500; mat <- matrix(runif(p * p, min = -1, max = 1), nrow = p, ncol = p);mat <- (mat + t(mat)) / 2; diag(mat) <- 1;sigma <- nearPD(mat, corr = TRUE, do2eigen = TRUE)$mat

#fix n 
n = 50 
p_seq <- floor(70*gamma_seq)
p_seq
p_seq <- c(0, p_seq)
partition = 0.7;test_error = NULL; result_sim1 <- NULL; repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL
sampled_cols <- 0 ; base_df <- data.frame(matrix(NA, 100, 0))

foreach(s = 1:100, .packages = c("caret","MASS")) %dopar% {
#for(s in 1 :100) {
  test_error = NULL;sampled_cols <- 0 ; base_df <- data.frame(matrix(NA, 100, 0));
for ( i in 2:21) {
  # don't need to sample randomly 
    p = p_seq[i]
    num_cols <- p - p_seq[i-1]
    remaining_cols <- setdiff(1:300, sampled_cols)
    new_cols <- sample(remaining_cols,num_cols, replace = FALSE)
    sampled_cols <- c(sampled_cols, new_cols)
    samp_data <- cbind(base_df, data[, as.vector(new_cols)],data[,ncol(data)])
    base_df <- cbind(base_df, data[, new_cols])
    
    trainIndex <- createDataPartition(data[, ncol(data)], p = partition, list = FALSE)
    trainIndex <- createDataPartition(samp_data[, ncol(samp_data)], p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y)#; pred <- predict_lda(train.x,mod) 
    #train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- c(test_error, mean(as.numeric(as.character(pred2))!=test.y))
   # repeated_test_error <- c(repeated_test_error,test_error)
}
  #result_sim1 <- rbind(result_sim1, test_error)
  result_sim1 <- test_error
  return(result_sim1)
} -> results

average_elements <- sapply(seq_along(results[[1]]), function(i) {
  mean(sapply(results, function(x) x[[i]]))
})

colMeans(result_sim1)
plot(p_seq[2:21]/100,average_elements, type = "b", xlab = expression(gamma), ylab = "Error Rate", main = "Mis-specified Model")
plot(p_seq[2:21]/100, colMeans(result_sim1), type = "b")
plot(p_seq[2:21], test_error, type = "b")
plot(p_seq/100, test_error)
  result_sim1 <- rbind(result_sim1,mean(repeated_test_error))
  #result_sim1 <- rbind(result_sim1,c(mean(repeated_train_error),mean(repeated_test_error)))
   gamma_result <- c(gamma_result, p/(nrow(train)))
  #train_row <- c(train_row, nrow(train))  repeated_train_error <- NULL
   repeated_test_error <- NULL
}
#foreach(i = 1:20, .packages = c("caret","MASS")) %dopar% {
for ( i in 1:20) {
  p <- p_seq[i]
  repeated_test_error <- NULL
  # Use foreach to replace the inner loop
    num_cols <- p - p_seq[i-1]
    sampled_columns <- sample(1:ncol(sigma)-1,num_cols, replace = FALSE)
    samp_data <- cbind(base_df, data[, as.vector(sampled_columns)],data[,ncol(data)])
    trainIndex <- createDataPartition(samp_data[, ncol(samp_data)], p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:(ncol(train)-1)]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:(ncol(test)-1)]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y)
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2)) != test.y)
    #repeated_test_error <- c(repeated_test_error,test_error)

  base_df <- samp_data
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train))
 # return(list(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
}#-> results

plot(gamma_result, result_sim1, type = "b")
df_mis<- cbind(gamma_result, result_sim1)
df_mis2 <- rbind(c(1,0.02222222),df_mis)
plot(df_mis2[,1],df_mis2[,2], type = "b")
write.csv(df_mis2, "mis_var3mean10_s200.csv", row.names = FALSE) #variance 3, mean 1,0 

result_sim1 <- sapply(results, `[[`, "result_sim1")
result_se <- sapply(results, `[[`, "result_se")
gamma_result <- sapply(results, `[[`, "gamma_result")
plot(gamma_result, result_sim1, type = "b", xlab = expression(gamma), ylab = "Error Rate", main = "Misspecified Model")
plot(gamma_result, result_se, type = "b")

#test error 

he <- read.csv("HE.csv", header = TRUE,row.names = 1 )
he
he <- as.data.frame(he)
############ only a subset of feature is discriminatory, mean #########################################

library(doParallel)
library(foreach)

p <- 300  
discriminant_features <- 200 
sigma <- diag(p)
diag(sigma) = 5
library(Matrix)
set.seed(123) # For reproducibility
mat <- matrix(runif(p * p, min = -1, max = 1), nrow = p, ncol = p);mat <- (mat + t(mat)) / 2; diag(mat) <- 1; sigma <- nearPD(mat, corr = TRUE, do2eigen = TRUE)$mat

mean1 <- c(runif(discriminant_features, 0.5, 1), rep(0, p - discriminant_features))
mean2 <- c(runif(discriminant_features, 1, 2), rep(0, p - discriminant_features))

# mean1 <- c(rep(1,p))
# mean2 <- c(rep(0,p))

group1 <- mvrnorm(50,mean1,sigma);
group2 <- mvrnorm(50,mean2,sigma)
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.105556, 2.5, length=10) 
gamma_seq <- c(gamma_seq1,gamma_seq2)
response_g1 <- rep(1,50); response_g2 <- rep(-1,50); response <- c(response_g1,response_g2)
data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))

# random_columns <- sample((1:ncol(data)-1), 100)
# # Set all values in the randomly selected columns to 0
# data[random_columns] <- lapply(data[random_columns], function(x) x * 0)

library(doParallell);library(foreach)
# Set up a parallel backend with the number of available cores
no_cores <- detectCores() - 1
registerDoParallel(cores=no_cores)
partition = 0.7;result_sim1 <- NULL; repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL; result_se <- NULL
# Use foreach to replace the outer loop
foreach(i = 1:20, .packages = c("caret","MASS")) %dopar% {
  p <- p_seq[i]
  repeated_test_error <- NULL
  for(ind in 1:100) {
    sampled_columns <- sample(1:(ncol(sigma)-1),p, replace = FALSE)
    samp_data <- cbind(data[, as.vector(sampled_columns)],data[,ncol(data)])
    trainIndex <- createDataPartition(samp_data[, ncol(samp_data)], p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y) #; pred <- predict_lda(train.x,mod) 
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2)) != test.y)
    repeated_test_error <- c(repeated_test_error,test_error)
  } 
  #result_sim1 <- rbind(result_sim1, mean(repeated_test_error))
  #result_se <- rbind(result_se, sd(repeated_test_error))
  #gamma_result <- c(gamma_result, p/(nrow(train)))
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train))
  return(list(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
}-> results
result_sim1 <- sapply(results, `[[`, "result_sim1")
result_se <- sapply(results, `[[`, "result_se")
gamma_result <- sapply(results, `[[`, "gamma_result")
plot(gamma_result, result_sim1, type = "b", xlab = expression(gamma), ylab = "Error Rate", main = "Misspecified Model")
plot(gamma_result, result_se, type = "b")




###############nois and redundant featuers ###############################################################################

delta = 1
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.1, 1, length=10) 
gamma_seq2 <- seq(1.02, 4, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7
noise_level = 6
result_sim1 <- NULL;repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL; train_row <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%  {
  #for (i in 1:20) {
  p = p_seq[i]
  n = nk_seq[i]
  repeated_test_error <- NULL
  for ( ind in 1: 80 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p);  sigma <- diag(p)
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    cols <- as.vector(sample(1:(ncol(data)-1), size = ceiling(0.40* ncol(data))  ))
    data[, cols] <- 0 
   #data[, cols] <- data[, cols] + matrix(rnorm(length(cols) * n*2, sd = noise_level), nrow = n*2, ncol = length(cols))
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train)) 
 return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
}-> results
plot(results[,3], results[,1], type = "b")

#redundant features
sig_4_10 <- results
sig_4_20 <- results
sig_4_30 <- results
sig_4_40 <- results
#noisy observation 
sig_1_15 <- results
sig_4_15 <- results
sig_6_15 <- results
write.csv(sig_4_10, "signal_4_10.csv", row.names = FALSE) 
write.csv(sig_4_20, "signal_4_20.csv", row.names = FALSE) 
write.csv(sig_4_30, "signal_4_30.csv", row.names = FALSE) 
#write.csv(sig_4_40, "signal_4_40.csv", row.names = FALSE) 

write.csv(sig_1_15, "signal_1_15.csv", row.names = FALSE) 
write.csv(sig_4_15, "signal_4_15.csv", row.names = FALSE) 
write.csv(sig_6_15, "signal_6_15.csv", row.names = FALSE) 

sig_4_10 <- read.csv("signal_4_10.csv");sig_4_20 <- read.csv("signal_4_20.csv");sig_4_30 <- read.csv("signal_4_30.csv")



plot(sig_4_10[,3],sig_4_10[,1], col= "darkred", ylim = c(0,0.5), type = "p" , pch =21, xlim =c(0,4),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "Redundant Features")
lines(sig_4_10[,3],sig_4_10[,1],col="darkred",lwd=1.2)
lines(sig_4_20[,3],sig_4_20[,1],col="darkblue",lwd=1.2)
points(sig_4_20[,3],sig_4_20[,1], pch=17, col = "darkblue")
points(sig_4_30[,3],sig_4_30[,1],pch=15,col="darkgreen")
lines(sig_4_30[,3],sig_4_30[,1], pch=19,col="darkgreen")
legend(x = 2, y = 0.5, legend = c("10%","20%","30%"), title = "Redundant Features Percentage",
       col = c("darkred", "darkblue", "darkgreen"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       cex = 1, 
       pch = c(21, 17, 15),
       bty = "n")

plot(sig_1_15[,3],sig_1_15[,1], col= "darkred", ylim = c(0,0.5), type = "p" , pch =21, xlim =c(0,4),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "Varying Noise")
lines(sig_1_15[,3],sig_1_15[,1],col="darkred",lwd=1.2)
lines(sig_4_15[,3],sig_4_15[,1],col="darkblue",lwd=1.2)
points(sig_4_15[,3],sig_4_15[,1], pch=17, col = "darkblue")
points(sig_6_15[,3],sig_6_15[,1],pch=15,col="darkgreen")
lines(sig_6_15[,3],sig_6_15[,1], pch=19,col="darkgreen")
legend(x = 2.5, y = 0.5, legend = expression(paste(sigma, " = 1"), paste(sigma, " = 4"), paste(sigma, " = 6")), title = "Noise Level",
       col = c("darkred", "darkblue", "darkgreen"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       cex = 1, 
       pch = c(21, 17, 15),
       bty = "n")



################Flippng Y lablel#######################################################################
#Flip x% of Y labelsd
delta = 1
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.1, 1, length=10) 
gamma_seq2 <- seq(1.02, 4, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
partition = 0.7
flip_percentage = 0.2
result_sim1 <- NULL;repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL; train_row <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%  {
  #for (i in 1:20) {
  p = p_seq[i]
  n = nk_seq[i]
  repeated_test_error <- NULL
  for ( ind in 1: 100 ) {
    mu1 <- rep(0,p) ; mu2 <- rep(0+delta,p);  sigma <- diag(p)
    group1 <- mvrnorm(n,mu1,sigma); group2 <- mvrnorm(n,mu2,sigma)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    rows <- as.vector(sample(1:(nrow(data)), size = ceiling(flip_percentage* nrow(data))))
    data$response[rows] <- -data$response[rows]
    #data$response <- as.factor(data$response)
    trainIndex <- createDataPartition(data$response, p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y); pred <- predict_lda(train.x,mod) 
    train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2))!=test.y)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train)) 
  return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
}-> results
plot(results[,3], results[,1], type = "b")

#redundant features
flip_05 <- results
flip_10 <- results
flip_20 <- results

write.csv(flip_05, "flip_05.csv", row.names = FALSE) 
write.csv(flip_10, "flip_10.csv", row.names = FALSE) 
write.csv(flip_20, "flip_20.csv", row.names = FALSE) 


plot(flip_05[,3],flip_05[,1], col= "darkred", ylim = c(0,0.5), type = "p" , pch =21, xlim =c(0,4),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "Varying Noise")
lines(flip_05[,3],flip_05[,1],col="darkred",lwd=1.2)
lines(flip_10[,3],flip_10[,1],col="darkblue",lwd=1.2)
points(flip_10[,3],flip_10[,1], pch=17, col = "darkblue")
points(flip_20[,3],flip_20[,1],pch=15,col="darkgreen")
lines(flip_20[,3],flip_20[,1], pch=19,col="darkgreen")
legend(x = 3, y = 0.5, legend = c("5%","10%","20%"), title = "Noise Level",
       col = c("darkred", "darkblue", "darkgreen"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       cex = 1, 
       pch = c(21, 17, 15),
       bty = "n")



#############non-normal###########################################################################################

#t distribution 
# mixture normal distribution 

p <- 300  
discriminant_features <- 200 ;mean1 <- c(rep(1,discriminant_features), rep(0, p - discriminant_features));mean2 <- c(rep(0,discriminant_features), rep(0, p - discriminant_features))
sigma <- diag(p)
diag(sigma) = 3
library(Matrix)
set.seed(123);mat <- matrix(runif(p * p, min = -1, max = 1), nrow = p, ncol = p);mat <- (mat + t(mat)) / 2; diag(mat) <- 1; sigma <- nearPD(mat, corr = TRUE, do2eigen = TRUE)$mat

mean1 <- rep(0,p)
mean2 <- rep(1,p)
group1 <- rmvt(n=50, sigma=sigma, df=30, delta=mean1)
group2 <- rmvt(n=50, sigma=sigma, df=30, delta=mean2)
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.105556, 2.5, length=10) 
gamma_seq <- c(gamma_seq1,gamma_seq2)
response_g1 <- rep(1,50); response_g2 <- rep(-1,50); response <- c(response_g1,response_g2)
data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))


result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL
# Use foreach to replace the outer loop
foreach(i = 1:20, .packages = c("caret","MASS"), .combine = 'rbind') %dopar% {
  p <- p_seq[i]
  # Use foreach to replace the inner loop
  library(doParallel)
  library(foreach)
  repeated_test_error <- foreach(ind = 1:300, .combine = c) %dopar% {
    sampled_columns <- sample(1:ncol(sigma)-1,p, replace = FALSE)
    samp_data <- cbind(data[, as.vector(sampled_columns)],data[,ncol(data)])
    trainIndex <- createDataPartition(samp_data[, ncol(samp_data)], p = partition, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y)
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2)) != test.y)
    test_error <- mean(as.numeric(as.character(pred2)) != test.y)
    # Return the test_error for each iteration
    return(test_error)
  }
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train))  # Make sure 'train' is defined correctly here
  return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
} -> results

plot(results[,3], results[,1], type = "b")
plot(gamma_result, result_se, type = "b")


#######################t-distribution########################################

nk_seq <- floor(seq(50, 100, length = 20)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

result_sim1 <- NULL; result_se <- NULL; gamma_result <- NULL
#Use foreach to replace the outer loop
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%  {
#for (i in 12:20) {
  p <- p_seq[i]
  n <- nk_seq[i]
  repeated_test_error <- NULL
for (ind in 1:80) {
    mean1 <- rep(0,p);  mean2 <- rep(1, p);
    #mean1 <- runif(p, 0,1);  mean2 <- runif(p, 0.5,1.5);
    sigma <- diag(p)
    sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- 0.3
    #sigma <- diag(p)
    #mat <- matrix(runif(p * p, min = -1, max = 1), nrow = p, ncol = p);mat <- (mat + t(mat)) / 2; diag(mat) <- 1;sigma <- as.matrix(nearPD(mat, corr = TRUE, do2eigen = TRUE)$mat)
    group1 <- rmvt(n=n, sigma=sigma, df=4, delta=mean1); group2 <- rmvt(n=n, sigma=sigma, df=4, delta=mean2)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y)
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2)) != test.y)
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- mean(repeated_test_error); result_se <- sd(repeated_test_error); gamma_result <- p/(nrow(train))
  #result_sim1 <- c(result_sim1,mean(repeated_test_error)); result_se <- c(result_se, sd(repeated_test_error));gamma_result <- c( gamma_result, p/(nrow(train)) )
  return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
} -> results

id_df1 <- results
id_df4 <- cbind(gamma_result,result_sim1)
id_df30 <- results

cs_df1 <- results
cs_df4 <- results
cs_df30 <- results

plot(gamma_result,result_sim1, type = "b")
abline(v= 1, col = "grey" ,lwd = 2)

result_sim1
#0.19625000 0.08958333 0.06015625 0.05551471 0.06145833 0.09166667 0.24506579 0.09343750 0.04077381 0.03660714


plot(results[,3], results[,1], main = "T-Distribution (Identity Covariance Matrix)", xlab = expression(gamma), ylab = "Error Rate", pch = 10, col = "darkred")
lines(results[,3], results[,1], lwd = 2, col = "darkred")
abline(v= 1, col = "grey" ,lwd = 2)
plot(gamma_result, result_se, type = "b")

results_tdist_0112 <- results
results_tdist_010515 <- results
results_tdist_010212 <- results
results_tdist_rho03 <- results
results_tdist_rho01 <- results
results_tdist_rho05 <- results
write.csv(results_tdist_010212, "tdist_identity_010212.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_tdist_0112, "tdist_identity_0112.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_tdist_010515, "tdist_identity_010515.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_tdist_rho03, "tdist_cs_rho03.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_tdist_rho05, "tdist_cs_rho05.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_tdist_rho01, "tdist_cs_rho01.csv", row.names = FALSE) #variance 3, mean 1,0 

write.csv(id_df1, "tdist_id_df1.csv", row.names = FALSE) #variance 3, mean 1,0
write.csv(id_df4, "tdist_id_df4.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(id_df30, "tdist_id_df30.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_tdist_rho01, "tdist_cs_rho01.csv", row.names = FALSE) #variance 3, mean 1,0 

write.csv(cs_df1, "cs_df1_03.csv", row.names = FALSE) #variance 3, mean 1,0
write.csv(cs_df4, "cs_df4_03.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(cs_df30, "cs_df30_03.csv", row.names = FALSE) #variance 3, mean 1,0 

results_tdist_rho01 <- read.csv("tdist_cs_rho01.csv")
results_tdist_rho03 <- read.csv("tdist_cs_rho03.csv")
results_tdist_rho05 <- read.csv("tdist_cs_rho05.csv")



plot(results_tdist[,3], results_tdist[,1], type = "b")

plot(results_tdist_010515[1:20,3], results_tdist_010515[1:20,1], ylim = c(0,0.5), main = "T-Distribution (Identity Covariance Matrix)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_tdist_010515[1:20,3], results_tdist_010515[1:20,1], lwd = 2, col = "darkred")
points(results_tdist_010212[1:20,3], results_tdist_010212[1:20,1], pch = 22, col = "darkgreen")
lines(results_tdist_010212[1:20,3], results_tdist_010212[1:20,1], lwd =2, col = "darkgreen")
points(results_tdist_0112[1:20,3], results_tdist_0112[1:20,1], pch = 23, col = "darkblue")
lines(results_tdist_0112[1:20,3], results_tdist_0112[1:20,1], lwd =2, col = "darkblue")
legend(x =2, y = 0.45, 
       legend = c(expression(mu[1] ~ "~ Unif(0, 1), "~mu[2] ~ "~Unif(0.5, 1.5)"), 
                  expression(mu[1] ~ "~ Unif(0, 1), "~mu[2] ~ "~Unif(0.2, 1.2)"),
                  expression(mu[1] ~ "~ Unif(0, 1), "~mu[2] ~ "~Unif(1, 2)")),
       col = c("darkred","darkgreen","darkblue"), bty = "n",
       lty = 1, 
       pch = c(19,22,23),
       cex = 0.9, 
       text.col = "black")


plot(results_tdist_rho01[,3],results_tdist_rho01[,1], col= "darkred", ylim = c(0,0.5), type = "l" ,xlim =c(0,4),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = " T-Distribution Compound Symmetric Covariance Matrix")
lines(results_tdist_rho03[,3],results_tdist_rho03[,1],col="darkgreen",lwd=1.2)
lines(results_tdist_rho05[,3],results_tdist_rho05[,1],col="darkblue",lwd=1.2)
points(results_tdist_rho01[,3],results_tdist_rho01[,1], pch=17, col = "darkred")
points(results_tdist_rho03[,3],results_tdist_rho03[,1],pch=15,col="darkgreen")
points(results_tdist_rho05[,3],results_tdist_rho05[,1], pch=19,col="darkblue")
legend(x = 2.5, y = 0.5, cex =1.2, legend = expression(paste(cov(x[i], x[j]), " = 0.1"),
                                                       paste(cov(x[i], x[j]), " = 0.3"), 
                                                       paste(cov(x[i], x[j]), " = 0.5")), 
       col = c("darkred", "darkgreen", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(17, 15, 19),
       bty = "n")

plot(id_df1[,3],id_df1[,1], col= "darkred", ylim = c(0,0.5), type = "l" ,xlim =c(0,4),lwd=1.2,xlab =expression(gamma),
     ylab="Error Rate", main = " T-Distribution Identity Covariance Matrix")
lines(id_df4[,1],id_df4[,2],col="darkgreen",lwd=1.2)
lines(id_df30[,3],id_df30[,1],col="darkblue",lwd=1.2)
points(id_df1[,3],id_df1[,1], pch=17, col = "darkred")
points(id_df4[,1],id_df4[,2],pch=15,col="darkgreen")
points(id_df30[,3],id_df30[,1], pch=19,col="darkblue")
legend(x = 2.5, y = 0.5, cex =1.2, legend = c(" df = 1",   " df = 4",  " df = 30"), 
       col = c("darkred", "darkgreen", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(17, 15, 19),
       bty = "n")


plot(cs_df1[,3],cs_df1[,1], col= "darkred", ylim = c(0.2,0.55), type = "l" ,xlim =c(0,4),lwd=1.2,xlab =expression(gamma),
     ylab="Error Rate", main = " T-Distribution Compound Symmetric Covariance Matrix")
lines(cs_df4[,3],cs_df4[,1],col="darkgreen",lwd=1.2)
lines(cs_df30[,3],cs_df30[,1],col="darkblue",lwd=1.2)
points(cs_df1[,3],cs_df1[,1], pch=17, col = "darkred")
points(cs_df4[,3],cs_df4[,1],pch=15,col="darkgreen")
points(cs_df30[,3],cs_df30[,1], pch=19,col="darkblue")
legend(x = 2.5, y = 0.54, cex =1.2, legend = c(" df = 1",   " df = 4",  " df = 30"), 
       col = c("darkred", "darkgreen", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(17, 15, 19),
       bty = "n")

###########poisson distribution#############################################################
sample_multi_poisson <- function(n, p, min, max) {
  lambda <- runif(p, min, max)
  samples <- matrix(nrow = n, ncol = p)
  for (i in 1:p) {
    samples[, i] <- rpois(n, lambda[i])
  }
  samples_df <- data.frame(samples)
  return(samples_df)
}

nk_seq <- floor(seq(50, 50, length = 30)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.01, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL

foreach(i = 1:30, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar%
 {
  p <- p_seq[i]
  n <- nk_seq[i]
  # Use foreach to replace the inner loop
  repeated_test_error <- NULL
  for (ind in 1:100) {
    group1 <- sample_multi_poisson(n,p,50,60); group2 <- sample_multi_poisson(n,p,60,70)
    response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
    data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
    trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
    train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
    test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
    mod <- lda_ginv(train.x,train.y)
    pred2 <- predict_lda(test.x,mod)
    test_error <- mean(as.numeric(as.character(pred2)) != test.y)
    # Return the test_error for each iteration
    repeated_test_error <- c(repeated_test_error,test_error)
  }
  result_sim1 <- mean(repeated_test_error)
  result_se <- sd(repeated_test_error)
  gamma_result <- p/(nrow(train))  # Make sure 'train' is defined correctly here
  return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
} -> results

plot(results[,3], results[,1], type = "b")

plot(results_pois[1:20,3], results_pois[1:20,1], ylim = c(0,0.5), main = "Poisson Distribution", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_pois[1:20,3], results_pois[1:20,1], lwd = 2, col = "darkred")
points(results_pois_2446[1:20,3], results_pois_2446[1:20,1], pch = 22, col = "darkgreen")
lines(results_pois_2446[1:20,3], results_pois_2446[1:20,1], lwd =2, col = "darkgreen")
points(results_pois_6778[1:20,3], results_pois_6778[1:20,1], pch = 23, col = "darkblue")
lines(results_pois_6778[1:20,3], results_pois_6778[1:20,1], lwd =2, col = "darkblue")
legend(x =2, y = 0.45, 
       legend = c(expression(lambda[1] ~ "~ Unif(6, 7), "~lambda[2] ~ "~Unif(7, 8)"), 
                  expression(lambda[1] ~ "~ Unif(2, 4), "~lambda[2] ~ "~Unif(4, 6)"),
                  expression(lambda[1] ~ "~ Unif(2, 3), "~lambda[2] ~ "~Unif(3, 4)")),
       col = c("darkblue","darkgreen","darkred"), bty = "n",
       lty = 1, 
       pch = c(23,22,19),
       cex = 0.8, 
       text.col = "black")
plot(gamma_result, result_se, type = "b")

results_pois <- results
results_pois_2446 <- results
results_pois_6778 <- results
write.csv(results_pois, "poisson_unif_23_34.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_pois_2446, "poisson_unif_24_46.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_pois_6778, "poisson_unif_67_78.csv", row.names = FALSE) #variance 3, mean 1,0 
plot(results_tdist[,3], results_tdist[,1], type = "b")





#############Dirichlet Distribution##################################################
library(gtools)

nk_seq <- floor(seq(50, 100, length = 10)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq
result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL
foreach(i = 1:10, .packages = c("caret","MASS","mvtnorm","Matrix","gtools"), .combine = 'rbind') %dopar%
  {
    p <- p_seq[i]
    n <- nk_seq[i]
    # Use foreach to replace the inner loop
    repeated_test_error <- NULL
    for (ind in 1:200) {
      m1 = rep(0, p); m2 = rep(1,p); 
      group1 <- rdirichlet(n, m2); group2 <- rdirichlet(n, m2); group1 <- data.frame(group1); group2 <- data.frame(group2)
      response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
      data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
      trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
      train <- data[trainIndex, ];  test <- data[-trainIndex, ]
      train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
      test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
      mod <- lda_ginv(train.x,train.y)
      pred2 <- predict_lda(test.x,mod)
      test_error <- mean(as.numeric(as.character(pred2)) != test.y)
      # Return the test_error for each iteration
      repeated_test_error <- c(repeated_test_error,test_error)
    }
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train))  
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
  } -> results2

plot(results2[,3], results2[,1], main = "Dirichlet Distribution", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results2[1:20,3], results2[1:20,1], lwd = 2, col = "darkred")

#abline(v= 1, col = "grey" ,lwd = 2)
plot(gamma_result, result_se, type = "b")

results_dir_01<- results
write.csv(results_dir, "dirichlet_01.csv", row.names = FALSE) #variance 3, mean 1,0 



#############Uniform Distribution###########################

sample_uniform <- function(n, p, min, max) {
  samples <- matrix(nrow = n, ncol = p)
  for (i in 1:p) {
    samples[, i] <- runif(n, min, max)
  }
  samples_df <- data.frame(samples)
  return(samples_df)
}

nk_seq <- floor(seq(50, 100, length = 30)) #training sample size, each group
nk_seq
n_full <- ceiling(nk_seq*2/(0.7))  #total dataset training + testing 
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix","gtools"), .combine = 'rbind') %dopar%
  {
    p <- p_seq[i]
    n <- nk_seq[i]
    # Use foreach to replace the inner loop
    repeated_test_error <- NULL
    for (ind in 1:200) {
      group1 <- sample_uniform(n,p,0, 0.5); group2 <- sample_uniform(n,p,0.1, 0.6);
      response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
      data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
      trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
      train <- data[trainIndex, ];  test <- data[-trainIndex, ]
      train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
      test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
      mod <- lda_ginv(train.x,train.y)
      pred2 <- predict_lda(test.x,mod)
      test_error <- mean(as.numeric(as.character(pred2)) != test.y)
      # Return the test_error for each iteration
      repeated_test_error <- c(repeated_test_error,test_error)
    }
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train))  # Make sure 'train' is defined correctly here
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
  } -> results

plot(results_unif_025075[,3], results_unif_025075[,1], ylim = c(0,0.5), main = "Uniform Distribution", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_unif_025075[1:20,3], results_unif_025075[1:20,1], lwd = 2, col = "darkred")
points(results_unif_007[,3], results_unif_007[,1], pch = 22, col = "darkgreen")
lines(results_unif_007[,3], results_unif_007[,1], lwd =2, col = "darkgreen")
points(results_unif_0106[1:20,3], results_unif_0106[1:20,1], pch = 23, col = "darkblue")
lines(results_unif_0106[1:20,3], results_unif_0106[1:20,1], lwd =2, col = "darkblue")
legend(x =2, y = 0.45, 
       legend = c("group 1 ~ Unif(0, 0.5), group 2 ~ Unif(0.25, 0.75)", 
                  "group 1 ~ Unif(0, 0.5), group 2 ~ Unif(0, 0.7)",
                  "group 1 ~ Unif(0, 0.5), group 2 ~ Unif(0.1, 0.6)"),
       col = c("darkred","darkgreen","darkblue"), bty = "n",
       lty = 1, 
       pch = c(19,22,23),
       cex = 0.8, 
       text.col = "black")
plot(results[,3], results[,1])

results_unif_025075 <- results
results_unif_007 <- results
results_unif_0106 <- results
write.csv(results_unif_005_025075, "unif_05_025075.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_unif_007, "unif_007.csv", row.names = FALSE) #variance 3, mean 1,0 
write.csv(results_unif_0106, "unif_0106.csv", row.names = FALSE) #variance 3, mean 1,0 
plot(results_tdist[,3], results_tdist[,1], type = "b")



##########log-normal#################################################

nk_seq <- floor(seq(20, 50, length = 20)) 
nk_seq
n_full <- ceiling(nk_seq*2/(0.7)) 
gamma_seq1 <- seq(0.05, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
#seq(0.01, 4, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)
p_seq
result_sim1 <- NULL;result_se <- NULL;gamma_result <- NULL
foreach(i = 10:20 ,.packages = c("caret","MASS","mvtnorm","Matrix","gtools"), .combine = 'rbind') %dopar%
  {
    p <- p_seq[i]
    n <- nk_seq[i]
    repeated_test_error <- NULL
    for (ind in 1:200) {
      mean1 <- rep(1,p); mean2 <- rep(3,p); sigma <- diag(p)
      group1 <- exp(rmvnorm(n, mean1, sigma = sigma)); group2 <- exp(rmvnorm(n, mean1, sigma = sigma))
      response_g1 <- rep(1,n); response_g2 <- rep(-1,n); response <- c(response_g1,response_g2)
      data <- rbind(group1,group2); data <- as.data.frame(cbind(data,response))
      trainIndex <- createDataPartition(data[, ncol(data)], p = 0.7, list = FALSE)
      train <- data[trainIndex, ];  test <- data[-trainIndex, ]
      train.x <- as.matrix(train[,1:ncol(train)-1]); train.y <- (train[,ncol(train)])
      test.x <- as.matrix(test[,1:ncol(test)-1]); test.y <- (test[,ncol(test)])
      mod <- lda_ginv(train.x,train.y)
      pred2 <- predict_lda(test.x,mod)
      test_error <- mean(as.numeric(as.character(pred2)) != test.y)
      # Return the test_error for each iteration
      repeated_test_error <- c(repeated_test_error,test_error)
    }
    result_sim1 <- mean(repeated_test_error)
    result_se <- sd(repeated_test_error)
    gamma_result <- p/(nrow(train))  
    return(data.frame(result_sim1 = result_sim1, result_se = result_se, gamma_result = gamma_result))
  } -> results
results_log_01 <- results
1   0.5020000 0.06938169        1.125
2   0.4938095 0.07326894        1.280

results_unif_007 <- results
results_unif_0106 <- results


plot(results_log_01[,3], results_log_01[,1], ylim = c(0,0.5), main = "Log-normal Distribution", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(results_log_01[1:20,3], results_log_01[1:20,1], lwd = 2, col = "darkred")
points(results_log_01[,3], results_log_01[,1], pch = 22, col = "darkgreen")
lines(results_log_01[,3], results_log_01[,1], lwd =2, col = "darkgreen")
points(results_unif_0106[1:20,3], results_unif_0106[1:20,1], pch = 23, col = "darkblue")
lines(results_unif_0106[1:20,3], results_unif_0106[1:20,1], lwd =2, col = "darkblue")
legend(x =2, y = 0.45, 
       legend = c("group 1 ~ Unif(0, 0.5), group 2 ~ Unif(0.25, 0.75)", 
                  "group 1 ~ Unif(0, 0.5), group 2 ~ Unif(0, 0.7)",
                  "group 1 ~ Unif(0, 0.5), group 2 ~ Unif(0.1, 0.6)"),
       col = c("darkred","darkgreen","darkblue"), bty = "n",
       lty = 1, 
       pch = c(19,22,23),
       cex = 0.8, 
       text.col = "black")
plot(results[,3], results[,1], type = "b", col = "darkred", lwd = 1.5, xlab = expression(gamma), ylab = "Error Rate", main = "Log-normal Distribution")

results_log <- results
results_unif_007 <- results
results_unif_0106 <- results








