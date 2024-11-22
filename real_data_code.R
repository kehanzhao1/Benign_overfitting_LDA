

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
library(ggplot2)
# Set up a parallel backend with the number of available cores
no_cores <- detectCores() - 1
registerDoParallel(cores=no_cores)


setwd("C:/Users/kehan/OneDrive/Desktop/research/real data/arcene")
train_path <- "C:/Users/kehan/OneDrive/Desktop/research/real data/arcene/ARCENE/arcene_train.data"
train_labels_path <- "C:/Users/kehan/OneDrive/Desktop/research/real data/arcene/ARCENE/arcene_train.labels"
test_path <- "C:/Users/kehan/OneDrive/Desktop/research/real data/arcene/ARCENE/arcene_test.data"

valid_path <- "C:/Users/kehan/OneDrive/Desktop/research/real data/arcene/ARCENE/arcene_valid.data"
valid_label_path <- "C:/Users/kehan/OneDrive/Desktop/research/real data/arcene/ARCENE/arcene_valid.labels"
param_path <- "C:/Users/kehan/OneDrive/Desktop/research/real data/arcene/ARCENE/arcene.param"

param <- readLines(param_path)
arcene_train <- read.table(train_path, header = TRUE)
arcene_valid<- read.table(valid_path, header = TRUE) # 99 x 1000
train_lab <- read.table(train_labels_path, header = TRUE)
valid_lab <- read.table(valid_label_path, header = TRUE)
arcene_valid <- as.data.frame(cbind(arcene_valid, valid_lab))
nrow(arcene_train) #99 
ncol(arcene_train) #10000
train <- as.data.frame(cbind(arcene_train, train_lab))
test <- arcene_valid
arcene_test <-as.data.frame(arcene_test)
#arcene_test2 <- createDataPartition(arcene_test[, ncol(arcene_test)], p = 0.5, list = FALSE)


############EXPLORATORY DATA ANALYSIS ##########################################
summary(arcene)
summary(train)
#distance measure of two group 


#measure deviation from multivariate normality Q-Q plot 
g1 <- train[train[,ncol(train)] == 1, 1:20]
g2 <- train[train[,ncol(train)] == -1, 1:20]
summary(g1)
summary(g2)
for (i in 1:20) {
  p <- ggplot(g1, aes(sample = g1[[i]])) + 
    stat_qq() + 
    ggtitle(paste("Q-Q plot for feature", i)) + 
    theme_minimal()
  
  print(p)
}
qq_plot <- function(data, feature_name) {
  p <- ggplot(data, aes(sample = data[[feature_name]])) + 
    stat_qq() +
    #ggtitle(paste("Q-Q plot for", feature_name)) + 
    theme_minimal()
  return(p)
}
plots <- lapply(names(g1)[1:20], function(feature) qq_plot(g1, feature))

library(gridExtra)
combined_plot <- do.call(grid.arrange, c(plots, ncol=4))

#Shapiro-Wilk Test with Bonferroni correction
p_values <- sapply(g1[1:20], function(column) shapiro.test(column)$p.value)

alpha <- 0.05
bonferroni_threshold <- alpha / length(p_values)

# Determine which features deviate from normality after correction
rejected_normality <- p_values < bonferroni_threshold
sum(rejected_normality)

# Mardia Test 
library(MVN)
result <- mvn(data = g1[, 1:20], mvnTest = "mardia")
summary(result)
result

index_seq <- seq(1, 10000, by = 20); mardia <- NULL
for (i in 1:length(index_seq)) {
  result <- mvn(data = train[,index_seq[i]:index_seq[i+1]], mvnTest = "mardia")
  mardia <- c(mardia, result$multivariateNormality$Result)
}

#Box M test for equality of covariance matrices

library(biotools)
boxm <- boxM(train[,1:20], train[,ncol(train)]) #covariances are different across groups

index_seq <- seq(1, 10000, by = 20); boxm <- NULL
for (i in 1:length(index_seq)) {
  result <- boxM(train[,index_seq[i]:index_seq[i+1]],train[,ncol(train)])
  boxm <- c(boxm, result$p.value)
}
length(boxm < 0.05)
#install.packages("corrplot")
library(corrplot)
# equality of covariance visual 

image(cov_group1, main="Group 1", axes=FALSE)
image(cov_group2, main="Group 2", axes=FALSE)

g1 <- train[train[,ncol(train)] == 1, 1:4]
g2 <- train[train[,ncol(train)] == -1, 1:4]
constant_columns <- colnames(train)[apply(train, 2, function(x) length(unique(x)) == 1)]
print(constant_columns)
g1_reduced <- g1[, !(colnames(g1) %in% constant_columns)]
g2_reduced <- g2[, !(colnames(g2) %in% constant_columns)]
cov_group1 <- cov(g1_reduced)
cov_group2 <- cov(g2_reduced)
cov_group1; cov_group2
boxM(train[,1:5],train[,ncol(train)]) # p-value = 0.03867



# mahalanobis distance 

maha_distance <- function(delta, sigma){
  sig_inverse <- ginv(sigma) 
  distance <- t(delta)%*%sig_inverse %*% delta
  return(distance)
}


g1full <- train[train[,ncol(train)] == 1,]
g2full <- train[train[,ncol(train)] == -1, ]
#total dataset training + testing 
gamma_seq <- seq(0.001, 0.9, length=20) 
gamma_seq2 <- seq(1.1, 2, length=20)
gamma_seq <- c(gamma_seq1, gamma_seq2)
p_seq <- ceiling(99 * gamma_seq)  #training sample size * gamma
p_seq
maha <- NULL; maha_result = data.frame(matrix(nrow = 30, ncol = 39))
for ( s in 1:30) {
  g1full <- g1full[, c(sample(10000), ncol(g1full))]
  g2full <- g2full[, c(sample(10000), ncol(g2full))]
  maha = NULL
for (i in 2:length(p_seq))  {
  p <- p_seq[i]
  err <- rep(0, 20)
  # Different means
  mu1 <- colMeans(g1full[,1:p])
  mu2 <- colMeans(g2full[,1:p])
  sigma <- (cov(g1full[,1:p]) + cov(g2full[,1:p]))/2
  maha =  c(maha, as.numeric(maha_distance(mu1-mu2,sigma)))
}
  maha_result[s,] <- sqrt(maha)
  
}
colMeans(maha_result)
plot(p_seq[2:length(p_seq)]/99,colMeans(maha_result), type = "b", lwd = 1.8,
     xlab = expression(gamma), ylab = "Mahalanobis Distance", main = "Mahalanobis Distance" )

##########INCREASE FEATURE orderly##############################################


ginv(arcene_train[,1:ncol(arcene_train)])

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
}
predict_lda <- function(newdata, model) {
  score <-  as.vector(t(model$coef) %*% t(newdata))  
  pred <- ifelse(score > as.numeric(model$thresh), 1, -1)
  return(pred)
}

# gamma_seq 
p_seq1 <- seq(1, 100, by= 5)
p_seq2 <- floor(seq(101, 5000, length = 100))
p_seq <- c(p_seq1,p_seq2)
# p_seq
# p_seq <- c(0, p_seq)
#partition = 0.7;test_error = NULL; result_sim1 <- NULL; repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL
# sampled_cols <- 0 ; base_df <- data.frame(matrix(NA, 100, 0))
foreach(s = 1:100, .packages = c("caret","MASS")) %dopar% {
#  for(s in 1 :100) {
  test_error = NULL ; train.y <-train[,ncol(train)]; test.y <- (test[,ncol(test)])  #;sampled_cols <- 0 ; base_df <- data.frame(matrix(NA, 100, 0));
  for ( i in 1:length(p_seq)) {
 # foreach(i = 1:length(p_seq), .packages = c("caret","MASS")) %dopar% {
    p = p_seq[i]
    #data <- cbind(train[,1:p], [,ncol(data)])
    #trainIndex <- createDataPartition(data[, ncol(data)], p = partition, list = FALSE); train <- data[trainIndex, ];  test <- data[-trainIndex, ]
    #train.x <- as.matrix(train[,1:(ncol(train)-1)])
    #train.x <- as.matrix([,1:(ncol(train)-1)]); train.y <- (train[,ncol(train)])
    train.x <- as.matrix(train[,1:p]) #; train.y <- as.matrix(data(data[,ncol(data)]))
    test.x <- as.matrix(test[,1:p]); 
    mod <- lda_ginv(train.x,train.y)#; pred <- predict_lda(train.x,mod) 
    #train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_lda(test.x,mod)
    test_error <- c(test_error, mean(as.numeric(as.character(pred2))!=test.y)) 
    #test_error <- mean(as.numeric(as.character(pred2))!=test.y)
   #result_sim1 <- rbind(result_sim1, test_error); result_sim1 <- test_error
    #return(test_error)
   }  #-> results

plot(p_seq, test_error, type = "b", ylab = "Error Rate", xlab = "Dimension", main ="Ordered")
abline(v = 99, col = "red")
lines(lowess(p_seq[ 1:53], test_error))


plot(p_seq[1:60]/99, test_error[1:60], lwd = 1.2, type = "l", ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data")
points(p_seq[1:60]/99, test_error[1:60], pch = 19)
abline(v = 1, col = "red")
lines(lowess(p_seq[ 20:60]/99, test_error[20:60]), col = "blue")
lines(lowess(p_seq[ 1:20]/99, test_error[1:20]), col = "blue")
df_order <- cbind(p_seq[1:60],test_error[1:60])
write.csv(df_order, "arcene_ordered_2000.csv", row.names = FALSE)
df_order <- read.csv("arcene_ordered_2000.csv")
plot(df_order$V1/99, df_order$V2, lwd = 1.2, type = "b", ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (Ordered Feature)")
df_order$V1 <- df_order$V1/99
plot(df_order$V1[1:38], df_order$V2[1:38], xlim = c(0,10),ylim = c() lwd = 1.2, type = "b", ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (Ordered Feature)")


##############Randomly Sample Columns##########################################
# gamma_seq 

p_seq <- floor(seq(2, 1000, length = 100))

p_seq1 <- seq(2, 100, by= 5)
p_seq2 <- floor(seq(101, 5000, length = 100))
p_seq <- c(0, p_seq1,p_seq2)
# p_seq
# p_seq <- c(0, p_seq)
#partition = 0.7;test_error = NULL; result_sim1 <- NULL; repeated_train_error <- NULL;repeated_test_error <- NULL; gamma_result <- NULL
sampled_cols <- 0 ; train.y = train[, ncol(train)]; test.y = test[, ncol(test)]
foreach(s = 1:30, .packages = c("caret","MASS")) %dopar% {
 # for(s in 1 :100) {
test_error = NULL;sampled_cols <- NULL #; base_df_train <- data.frame(matrix(NA, 100, 0));base_df_test <- data.frame(matrix(NA, 100, 0));
for ( i in 2:length(p_seq)) {
  p = p_seq[i]
  num_cols <- p - p_seq[i-1]
  remaining_cols <- setdiff(1:10000, sampled_cols)
  new_cols <- sample(remaining_cols,num_cols, replace = FALSE)
  sampled_cols <- c(sampled_cols, new_cols)
  train.x <- train[, sampled_cols]
  #base_df_train <- train.x
  test.x <- test[,sampled_cols]; 
  mod <- lda_ginv(train.x,train.y)#; pred <- predict_lda(train.x,mod) 
  #train_error <- mean(as.numeric(as.character(pred))!=train.y)
  #pred2 <- predict(mod2, test)$class
  pred2 <- predict_lda(test.x,mod)
  test_error <- c(test_error, mean(as.numeric(as.character(pred2))!=test.y))
  #repeated_test_error <- c(repeated_test_error,test_error)
}
#result_sim1 <- rbind(result_sim1, test_error); result_sim1 <- test_error
return(test_error)
} -> results

trial1 <- test_error
trial2 <- test_error
trial3 <- test_error
plot(p_seq[2:length(p_seq)]/99, test_error, type = "b", ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (Randomly Sample Features)")
abline(v = 1, col = "red")
lines(lowess(p_seq[2:length(p_seq)]/99, test_error,))
write.csv(df_order, "arcene_ordered_2000.csv", row.names = FALSE)

averages <- sapply(1:120, function(i) {
  mean(sapply(results, function(lst) lst[i]))
})
plot(p_seq[2:length(p_seq)]/99, averages, type = "l",lwd = 1.2, ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (Randomly Sample Features)")
points(p_seq[2:length(p_seq)]/99, averages, pch = 19)
abline(v = 1, col = "red")
lines(lowess(p_seq[2:length(p_seq)]/99, test_error,))
dfrandom =  cbind(p_seq[2:length(p_seq)]/99, averages)
write.csv(dfrandom, "arcene_random.csv", row.names = FALSE)

df_random <- read.csv("arcene_random.csv")

############DLDA##################################################################
# apply some dimension reduction 
constant_columns <- colnames(train)[apply(train, 2, function(x) length(unique(x)) == 1)]
print(constant_columns)
train_reduced <- train[, !(colnames(train) %in% constant_columns)]
test_reduced <- test[, !(colnames(test) %in% constant_columns)]

invert_diagonal <- function(mat, tiny_constant = 1e-2) {
  diag_elements <- diag(mat)
  diag_elements[diag_elements == 0] <- tiny_constant
  inverted_elements <- 1 / diag_elements
  new_matrix <- diag(inverted_elements)
  return(new_matrix)
}

dlda_funct <- function(x, y) {
  n <- length(y)
  p <- ncol(x)
  mu <- matrix(0, nrow = p, ncol =2)
  S <- matrix(0, nrow = p, ncol = p)
  if ( p == 1) {
    mu[, 1] <- mean(x[y == 1, ])
    mu[, 2] <- mean(x[y == -1, ])
    S <- diag(cov(x))
  }
  else {
    mu[, 1] <- colMeans(x[y == 1, ])
    mu[, 2] <- colMeans(x[y == -1, ])
    S <- diag(cov(x))
    S <- diag(S)
  }
  #S_inv <- solve(S)
  S_inv <- invert_diagonal(S)
  dif_vec <- t(t(as.vector(mu[, 1] - mu[, 2])))
  sum_vec <- t(t(as.vector(mu[, 1] + mu[, 2])))
  coef <- S_inv %*% dif_vec
  thresh <- (1/2) * t(coef) %*% sum_vec
  return(list(coef = coef, thresh = thresh, prec_matrix = S_inv, group_mean = mu))
}
predict_dlda <- function(newdata, mod) {
  # newdata <- as.matrix(newdata)
  coef <- mod$coef
  score <- newdata %*% coef
  pred <- ifelse(score > as.numeric(mod$thresh), 1, -1)
  return(pred)
}
p_seq <- floor(seq(2, 1000, length = 100))
p_seq1 <- seq(2, 100, by= 10)
p_seq2 <- floor(seq(101, 1000, length = 50))
p_seq <- c(0, p_seq1,p_seq2)
sampled_cols <- 0 ; train.y = train[, ncol(train)]; test.y = test[, ncol(test)]
foreach(s = 1:k, .packages = c("caret","MASS")) %dopar% {
  # for(s in 1 :100) {
  test_error = NULL;sampled_cols <- NULL  #; base_df_train <- data.frame(matrix(NA, 100, 0));base_df_test <- data.frame(matrix(NA, 100, 0));
  for ( i in 2:length(p_seq)) {
    p = p_seq[i]
    num_cols <- p - p_seq[i-1]
    remaining_cols <- setdiff(1:ncol(train_reduced), sampled_cols)
    new_cols <- sample(remaining_cols,num_cols, replace = FALSE)
    sampled_cols <- c(sampled_cols, new_cols)
    train.x.reduced <- train_reduced[, sampled_cols]
    test.x.reduced <- test_reduced[,sampled_cols]; 
    mod <- dlda_funct(train.x.reduced,train.y)#; pred <- predict_lda(train.x,mod) 
    #train_error <- mean(as.numeric(as.character(pred))!=train.y)
    #pred2 <- predict(mod2, test)$class
    pred2 <- predict_dlda(as.matrix(test.x.reduced),mod)
    test_error <- c(test_error, mean(as.numeric(as.character(pred2))!=test.y))
    #repeated_test_error <- c(repeated_test_error,test_error)
  }
  #result_sim1 <- rbind(result_sim1, test_error); result_sim1 <- test_error
  return(test_error)
} -> results
averages <- sapply(1:120, function(i) {
  mean(sapply(results, function(lst) lst[i]))
})

averages2 <- sapply(1:60, function(i) {
  mean(sapply(results, function(lst) lst[i]))
})
plot(p_seq[2:length(p_seq)]/99, averages2, type = "l",lwd = 1.2, ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (Randomly Sample Features)")
points(p_seq[2:length(p_seq)]/99, averages2, pch = 19)
abline(v = 1, col = "red")
lines(lowess(p_seq[2:length(p_seq)]/99, test_error,))
dfrandom =  cbind(p_seq[2:length(p_seq)]/99, averages)
write.csv(dfrandom, "arcene_random.csv", row.names = FALSE)

dfrandom2 =  cbind(p_seq[2:length(p_seq)]/99, averages)
plot(dfrandom2[,1], dfrandom2[,2], type = "l",lwd = 1.2, ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (DLDA)")
points(dfrandom2[,1], dfrandom2[,2], pch = 19)
write.csv(dfrandom2, "arcene_dlda.csv", row.names = FALSE)
write.csv(results, "arcene_dlda_raw.csv", row.names = FALSE)
#[18,]  0.87878788 0.4424242


########PCA Feature selection#######################
train.x <- train[,1:(ncol(train)-1)]; train.y = train[,ncol(train)]
pca_result <- prcomp(train.x, center = TRUE) 
pc_scores <- as.data.frame(pca_result$x[,1:98])
test.x <- test[,1:(ncol(test)-1)]; test.y = test[,ncol(test)]
test_scores <- as.data.frame(scale(test.x, center = pca_result$center, scale = pca_result$scale) %*% pca_result$rotation[, 1:98])
test_error <- NULL
for ( i in 2:98) {
  mod <- lda_ginv(pc_scores[,1:i],train.y)#;
  #pred <- predict_lda(train.x,mod) 
  #train_error <- mean(as.numeric(as.character(pred))!=train.y)
  #pred2 <- predict(mod2, test)$class
  pred2 <- predict_lda(test_scores[,1:i],mod)
  test_error <- c(test_error, mean(as.numeric(as.character(pred2))!=test.y))
}
dfPCA =  cbind(c(2:98)/99, test_error)
write.csv(dfPCA, "arcene_PCA.csv", row.names = FALSE)
df_reduced = dfPCA[seq(1, nrow(dfPCA), by = 5), ]
plot(c(2:98)/99, test_error, xlab = expression(gamma), ylab = "Error Rate", main = "Feature Selection (PCA)")

plot(ceiling(df_reduced[,1]*99), df_reduced[,2], xlab ="Principal Components", pch = 16, ylim = c(0.1, 0.3),ylab = "Error Rate", main = "Feature Selection (PCA)", )
lines(lowess(ceiling(df_reduced[,1]*99), df_reduced[,2]), lwd = 1.8, col = "darkblue")



########PSEUDO-LDA##############################################################################
train_reduced.x <- train_reduced[, -1]
test_reduced.x <- test_reduced[, -1]
p_seq1 <- floor(seq(2, 100, length= 10))
p_seq2 <- floor(seq(101, 1000, length = 20))
p_seq <- c(p_seq1,p_seq2)
p_seq <- floor(seq(2, 1000, length = 30))
error <- NULL
for (i in 1:length(p_seq)){
  p = p_seq[i]
  train.x <- train_reduced.x[,1:p]
  test.x <- test_reduced.x[,1:p]
  mod <- lda_ginv(train.x,train.y)
  pred2 <- predict_lda(test.x,mod)
  error <- c(error, mean(as.numeric(as.character(pred2))!=test.y))
}
df_lda <- cbind(p_seq/99,error)
plot(df_lda[,1], df_lda[,2], ylim = c(0.1,0.6), type = "b")
abline(v=1)



##########High-DIM DA############################################################################

train_reduced.x <- train_reduced[, -1]
test_reduced.x <- test_reduced[, -1]
p_seq1 <- floor(seq(2, 100, length= 20))
p_seq2 <- floor(seq(101, 1000, length = 30))
p_seq <- floor(seq(2, 1000, length = 30))
p_seq <- c(p_seq1,p_seq2)
#foreach(s = 1:10, .packages = c("caret","MASS", "HiDimDA")) %dopar% {
 # error = NULL;
  #cols = sample(ncol(train_reduced.x))
  #train_reduced.x <- train_reduced.x[, cols]
  #test_reduced.x <- test_reduced.x[, cols]
error <- NULL
for (i in 1:length(p_seq)){
  p = p_seq[i]
  dlda <- Dlda(train_reduced.x[,1:p], as.factor(train.y), VSelfunct = "none")
  pred <- predict(dlda, test_reduced.x[1:p], grpcodes = levels(as.factor(test.y)))
  error <- c(error, mean(as.numeric(as.character(pred$class))!=test.y))
}
#return(error)
#} -> results
#averages <- sapply(1:50, function(i) {
 # mean(sapply(results, function(lst) lst[i]))
#})
plot(p_seq[2:length(p_seq)]/99, error, pch = 1)
df_dlda2 <- cbind(p_seq[2:length(p_seq)]/99, averages)
df_dlda3 <- cbind(p_seq[1:length(p_seq)]/99, error)
plot(dfrandom2[,1], dfrandom2[,2], type = "l",lwd = 1.2, ylab = "Error Rate", xlab = expression(gamma), main ="Arcene Data (DLDA)")
points(dfrandom2[,1], dfrandom2[,2], pch = 19)

plot(df_dlda3[,1], df_dlda3[,2])



######Slda########################################

slda <- Dlda(train.x[,1:10], as.factor(train.y), VSelfunct = "none")

pred <- predict(slda, test.x, grpcodes = levels(as.factor(test.y)))
mean(as.numeric(as.character(pred$class))!=test.y)
dlda <- Slda(train_reduced.x[,1:p], as.factor(train.y), VSelfunct = "none")
pred <- predict(dlda, test_reduced.x[1:p], grpcodes = levels(as.factor(test.y)))

train_reduced.x <- train_reduced[, -1]
test_reduced.x <- test_reduced[, -1]
p_seq1 <- floor(seq(2, 100, length= 20))
p_seq2 <- floor(seq(101, 1000, length = 30))
p_seq <- c(0, p_seq1,p_seq2)
p_seq <- floor(seq(2, 1000, length = 30))
#foreach(s = 1:10, .packages = c("caret","MASS", "HiDimDA")) %dopar% {
  error = NULL;
 # cols = sample(ncol(train_reduced.x))
  #train_reduced.x <- train_reduced.x[, cols]
  #test_reduced.x <- test_reduced.x[, cols]
  for (i in 1:length(p_seq)){
    p = p_seq[i]
    slda <- Slda(train_reduced.x[,1:p], as.factor(train.y), VSelfunct = "none")
    pred <- predict(slda, test_reduced.x[1:p], grpcodes = levels(as.factor(test.y)))
    error <- c(error, mean(as.numeric(as.character(pred$class))!=test.y))
  }
  return(error)
#} -> results2
df_slda <- cbind(p_seq/99, error)
plot(df_slda[,1], df_slda[,2])
  
averageslda <- sapply(1:50, function(i) {
  mean(sapply(results2, function(lst) lst[i]))
})
plot(p_seq[2:length(p_seq)]/99, averageslda, pch = 1)
plot(p_seq[2:length(p_seq)]/99, averageslda, pch = 1)

plot(p_seq[2:length(p_seq)]/99,error, pty = 19)
lines(p_seq[2:length(p_seq)]/99,error, pty = 19)


##MLDA#############################################################

train_reduced.x <- train_reduced[, -1]
test_reduced.x <- test_reduced[, -1]
p_seq1 <- floor(seq(2, 100, length= 20))
p_seq2 <- floor(seq(101, 1000, length = 30))
p_seq <- c(p_seq1,p_seq2)
p_seq <- floor(seq(2, 1000, length = 30))
error_mlda = NULL;
for (i in 1:length(p_seq)){
  p = p_seq[i]
  mlda <- Mlda(train_reduced.x[,1:p], as.factor(train.y), VSelfunct = "none")
  pred <- predict(mlda, test_reduced.x[1:p], grpcodes = levels(as.factor(test.y)))
  error_mlda <- c(error_mlda, mean(as.numeric(as.character(pred$class))!=test.y))
}
return(error_mlda)

df_mlda <- cbind(p_seq/99, error_mlda)
plot(df_mlda[,1],df_mlda[,2] )

###visualization############

df_total <- data.frame(df_lda, dlda = df_dlda3[,2], slda = df_slda[,2], mlda = df_mlda[,2])
write.csv(df_total, "ARCENE_combined_methods.csv")
df_total2 <- df_total[seq(2, nrow(df_total), by = 3), ]
plot(df_total2[,1],df_total2[,2], type = "l", col = "darkred", lwd = 1.8, ylim = c(0.1, 0.6), xlab = expression(gamma), ylab = "Error Rate", main = "Comparison of Error Rate")
lines(df_total2[,1], df_total2[,3], col = "darkblue", lwd = 1.8)
lines(df_total2[,1], df_total2[,4], col = "orange", lwd = 1.8)
lines(df_total2[,1], df_total2[,5], col = "darkgreen", lwd = 1.8)

points(df_total2[,1], df_total2[,2], col = "darkred", pch = 19 )
points(df_total2[,1], df_total2[,3], col = "darkblue", pch = 19)
points(df_total2[,1], df_total2[,4], col = "orange", pch = 19)
points(df_total2[,1], df_total2[,5], col = "darkgreen", pch = 19)
abline(v = 1, col = "red")

legend("topright", legend = c("Pseudo-LDA", "DLDA", "SLDA", "MLDA"), 
       col = c("darkred", "darkblue", "orange", "darkgreen"), 
       lty = 1, lwd = c(1.2, 1.8, 1.8, 1.8), 
       pch = c(19, 19, 19), cex = 1,
       bty = "n")


######BOOTstrap population param and apply theory #############################################

# Generating some example multivariate data
library(MASS)
library(parallel)

bootdata_g1 <- train[train[,ncol(train)] == 1, 1:100]
bootdata_g2 <- train[train[,ncol(train)] == -1, 1:100]
bootdata_tot <-  train[, 1:100]
n1 <- 43; n2 <- 56
write.csv(dfrandom, "arcene_random.csv", row.names = FALSE)

set.seed(123)
#n1 <- 56; n2 <- 43
# Bootstrapping with parallel processing
B <- 2000; mean_results <- NULL
foreach(s = 1:B, .packages = c("caret","MASS")) %dopar% {
 #for (s in 1 :1000) {
  sample_indices <- sample(row.names(bootdata_g1), n1, replace = TRUE)
  bootstrap_sample <- bootdata_g1[sample_indices, ]
  means <- colMeans(bootstrap_sample)
  mean_results <- rbind(mean_results, means)
  return(mean_results)
} -> means

g1_colmeans<- sapply(1:100, function(i) {
  mean(sapply(means, function(lst) lst[[i]]))
})

colMeans(bootdata_g1)

mean_g12000 <- colMeans(mean_results)
mean_g12000
write.csv((mean_g12000), "mean_g1.csv", row.names = FALSE)


B <- 2000; mean_results <- NULL
foreach(s = 1:B, .packages = c("caret","MASS")) %dopar% {
    #for(s in 1 :1000) {
  sample_indices <- sample(row.names(bootdata_g2), n2, replace = TRUE)
  bootstrap_sample <- bootdata_g2[sample_indices, ]
  means <- colMeans(bootstrap_sample)
  mean_results <- rbind(mean_results, means)
  return(mean_results)
} -> means

g2_colmeans<- sapply(1:100, function(i) {
  mean(sapply(means, function(lst) lst[[i]]))
})
colMeans(bootdata_g2)
mean_g22000 <- colMeans(mean_results)
mean_g22000
write.csv((mean_g12000), "mean_g1.csv", row.names = FALSE)

write.csv(g2_colmeans, "g12_100.csv", row.names = FALSE)
write.csv(g1_colmeans, "g1_100.csv", row.names = FALSE)

#covariance matrix 
B <- 1000; cov_results <- list()
foreach(s = 1:B, .packages = c("caret","MASS")) %dopar% {
  #for(s in 1 :1000) {
  sample_indices <- sample(row.names(bootdata_tot), 99, replace = TRUE)
  bootstrap_sample <- bootdata_tot[sample_indices, ]
  cov <- cov(bootstrap_sample)
  cov_results[[length(cov_results)+1]] <- cov
  return(cov_results)
} -> covs

cov(bootdata_tot)

matrix_dim <- dim(covs[[1]][[1]]) 
avg_matrix <- matrix(0, nrow=matrix_dim[1], ncol=matrix_dim[2])
for (i in 1:1000) {
  avg_matrix <- avg_matrix + covs[[i]][[1]]
}
avg_matrix <- avg_matrix / 1000

write.csv(avg_matrix, "cov100_bootstrap.csv", row.names = FALSE)


wang_error3 <- function(maha,  y) {
  # group1_value <- (maha)*(sqrt(1-y)) /(2*sqrt(maha + 4*y))   
  group2_value <- (maha) /(2*sqrt(maha + 4*y))
  rate <- pnorm(-(sqrt(1-y))*group2_value)
  return(rate)
}

maha_distance <- function(delta, sigma){
  sig_inverse <- ginv(sigma) 
  distance <- t(delta)%*%sig_inverse %*% delta
  return(distance)
}
delta = g1_colmeans - g2_colmeans 
maha = maha_distance(delta, avg_matrix)




#total dataset training + testing 
gamma_seq <- seq(0.01, 1, length=20) 
gamma_seq
p_seq <- c(1, seq(3, 99))  #training sample size * gamma
p_seq
gamma1 <- NULL;error_rates <- rep(0, 98); result = NULL;
g1_row <- row.names(train[train[,ncol(train)] == 1, ]);g2_row <- row.names(train[train[,ncol(train)] == -1, ])
for ( s in 1:30){
  error_rates <- rep(0, 98); sampled_cols = NULL
for (i in 2:length(p_seq)) {
  p <- p_seq[i];
  num_cols <- p - p_seq[i-1]
  remaining_cols <- setdiff(1:10000, sampled_cols)
  new_cols <- sample(remaining_cols,num_cols, replace = FALSE)
  sampled_cols <- c(sampled_cols, new_cols)
  trainsamp <- train[, sampled_cols];
  g1_samp <- trainsamp[g1_row, ]
  g2_samp <- trainsamp[g2_row, ]
  #samp_data <- bootdata_tot[,1:p]
  # sigma <- cov(samp_data);  
  #g1_samp <- train[train[,ncol(train)] == 1, 1:100]
  #sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.3
  #sigma = rho^abs(outer(1:p, 1:p, "-"))
  delta = colMeans(g1_samp) - colMeans(g2_samp)
  sigma = (cov(g1_samp) +cov(g2_samp))/2
  maha = as.numeric(maha_distance(delta,sigma))
  error_rates[i] <- wang_error3(maha, p/99)
}
  result <- rbind(result, error_rates)
}
write.csv(result, "theoryARCENE.csv", row.names = FALSE)
plot(p_seq[-1]/99, colMeans(result[,-1]), type = "p",pch = 19, xlab = expression(gamma), col = "darkblue",ylab = "Error Rate", main = "Theoretical Error Rate in Underparameterized Regime")
lines(p_seq[-1]/99, colMeans(result[,-1]), type = "l",lty = 1, lwd =1.2, col = "darkblue")



set.seed(123)
n1 <- 56; n2 <- 43
# Bootstrapping with parallel processing
B <- 1000
num_cores <- detectCores() - 1  # leave one core free
# Start a cluster
cl <- makeCluster(num_cores)

# Split B into chunks for each core
bootstrap_chunks <- splitIndices(B, num_cores)

# Bootstrap function
bootstrap_funcG1 <- function(indexes) {
  means <- matrix(0, length(indexes), ncol(bootdata_g1))
 # covs <- vector("list", length(indexes))
  for (i in seq_along(indexes)) {
    sample_indices <- sample(1:56, 43, replace = TRUE)
    bootstrap_sample <- bootdata_g1[sample_indices, ]
    means[i, ] <- colMeans(bootstrap_sample)
    #covs[[i]] <- cov(bootstrap_sample)
  }
  list(means = means)
}
# Export necessary data and functions to the cluster nodes
clusterExport(cl, c("bootdata_g1", "bootstrap_funcG1"))
# Using parLapply to run the bootstrapping in parallel
results <- parLapply(cl, bootstrap_chunks, bootstrap_funcG1)
# Stop the cluster
stopCluster(cl)

# Combining results
bootstrap_means <- do.call(rbind, lapply(results, `[[`, "means"))
bootstrap_covs <- do.call(c, lapply(results, `[[`, "covs"))

# Approximate true mean and covariance
approx_mean <- colMeans(bootstrap_means)
approx_cov <- Reduce("+", bootstrap_covs) / B

print(approx_mean)
print(approx_cov)



