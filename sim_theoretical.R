library(MASS)
library(caret)
library(rda) #Guo et al., 2013
library(discrim) # fisher and Guo 
library(rlda)
library(dplyr)
library(mvtnorm)
library(ggplot2)
library(pracma)
library(HiDimDA)
library(MLmetrics)
library(Rdimtools)

library(MASS)
library(ggplot2)

# Theoretical error rate for LDA
lda_error_rate <- function(mu1, mu2, sigma, xbar1, xbar2, s_inv) {
  numerator1 <- -t(mu1 - 0.5*(xbar1+xbar2)) %*%s_inv %*% (xbar1 - xbar2)
  numerator2 <- t(mu2 - 0.5*(xbar1+xbar2)) %*%s_inv %*% (xbar1 - xbar2)
  denominator <- sqrt(t(xbar1 - xbar2) %*% s_inv %*% sigma %*% s_inv %*% (xbar1 - xbar2))
  rate <- 0.5*pnorm(numerator1/denominator) + 0.5*pnorm(numerator2/denominator)
  return(rate)
}



#reducing the calculation 
n_seq1 <- floor(seq(50, 100, length=25))
n_seq1
gamma_seq1 <- seq(0.1, 2, length=25) 
gamma_seq1
p_seq1 <- ceiling(n_seq1*2 * gamma_seq1)
p_seq1

delta = 2
delta = 0.5
delta = 1
error_rates1 <- numeric(length(n_seq1))
gamma1 <- numeric(length(n_seq1))
groupmean1 = 0 - delta/2
groupmean2 = delta/2
for (i in 1:25) {
  n <- n_seq1[i]
  p <- p_seq1[i]
  err <- rep(0, 20)
  # Different means
  mu1 <- rep(groupmean1, p)
  mu2 <- rep(groupmean2, p)
  # Common covariance matrix- Identity 
  sigma <- diag(p)
  #sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.3
  for ( s in 1: 80) {
    #sample 50 times and average the LDA error rates 
    group1 <- mvrnorm(n,mu1,sigma)
    group2 <- mvrnorm(n,mu2,sigma)
    xbar1 <- colMeans(group1)
    xbar2 <- colMeans(group2)
    s_inv <-  ginv((cov(group1) + cov(group2))/2)
    #s_inv <- solve(cov(rbind(group1, group2)))
    # Compute error rate
    err[s] <- lda_error_rate(mu1, mu2, sigma, xbar1, xbar2, s_inv)
  } 
  error_rates1_05[i] <- mean(err)
  gamma1[i] <- p / (2*n)
}
delta_05 <- cbind(gamma1, error_rates1_05)
delta_1 <- cbind(gamma1, error_rates1_05)
delta_2 <- cbind(gamma1, error_rates1_05)
df_theory1_save <- cbind(delta_05,delta_1,delta_2)
write.csv(df_theory1_save, "identity_delta_theoretical", row.names = FALSE)


plot(delta_05[1:25,1],delta_05[1:25,2], type = "l" , ylim=c(0,0.4),xlim =c(0,2),lwd=1.2,xlab =expression(gamma),ylab="Error Rate",main= "Identity Covariance Matrix")
lines(delta_1[1:25,1],delta_1[1:25,2],col="darkred",lwd=1.2)
lines(delta_2[1:25,1],delta_2[1:25,2],col="darkblue",lwd=1.2)
points(delta_05[1:25,1],delta_05[1:25,2], pch=4)
points(delta_1[1:25,1],delta_1[1:25,2], pch=10,col="darkred")
points(delta_2[1:25,1],delta_2[1:25,2], pch=7,col="darkblue")
legend(x = 1.3, y = 0.4, cex =1.2, legend = expression(paste(delta, " = 0.5"), paste(delta, " = 1"), paste(delta, " = 2")), 
       col = c("black", "darkred", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(4, 10, 7),
       bty = "n")


# with feature correlation 

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

delta = 2
groupmean1 = 1
groupmean2 = 0
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

gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)
#for (i in 1:20) {
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar% {
  n <- nk_seq[i]
  p <- p_seq[i]
  err <- rep(0, 20)
  # Different means
  mu1 <- rep(groupmean1, p)
  mu2 <- rep(groupmean2, p)
  # Common covariance matrix- Identity 
  sigma <- diag(p)
  sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.1
  for ( s in 1: 50) {
    #sample 50 times and average the LDA error rates 
    group1 <- mvrnorm(n,mu1,sigma)
    group2 <- mvrnorm(n,mu2,sigma)
    xbar1 <- colMeans(group1)
    xbar2 <- colMeans(group2)
    s_inv <-  ginv((cov(group1) + cov(group2))/2)
    #s_inv <- solve(cov(rbind(group1, group2)))
    # Compute error rate
    err[s] <- lda_error_rate(mu1, mu2, sigma, xbar1, xbar2, s_inv)
  } 
  error_rates <- mean(err)
  gamma1 <- p / (2*n)
  return(data.frame(gamma = gamma1, result = error_rates ))
} -> results

plot(results[,1], results[,2])
theoretical_rho03 <- results
theoretical_rho01 <- results
theoretical_rho05 <- results


theoretical_rho05  =  cbind(gamma1, error_rates1_05)
theoretical_rho03  =  cbind(gamma1[-11], error_rates1_05[-11])
theoretical_rho07  =  cbind(gamma1[-11], error_rates1_05[-11])


theoretical_rho05  =  cbind(gamma1[-11], error_rates1_05[-11])
theoretical_rho03  =  cbind(gamma1[-11], error_rates1_05[-11])
theoretical_rho07  =  cbind(gamma1[-11], error_rates1_05[-11])
df_the <- cbind(theoretical_rho05, theoretical_rho03, theoretical_rho01)
write.csv(df_the, "theoretical_compoundsymmetric.csv", row.names = FALSE)

plot(gamma1[-11], error_rates1_05[-11])

plot(theoretical_rho03[,1], theoretical_rho03[,2],  type = "b", pch = 20,col = "palegreen4",
     xlab = "Dimension",ylab = "Testing Error",main = "Dimension vs Testing error (isotropic feature)", ylim = c(0,0.5))
points(theoretical_rho05[,1], theoretical_rho05[,2],  type = "b", pch = 20,col = "purple")
points(theoretical_rho07[,1], theoretical_rho07[,2],  type = "b", pch = 20,col = "orange")
legend(x = 3, y  =0.4, legend=c(expression(paste(rho,"= 0.1")),expression(paste(rho, "= 0.3")),expression(paste(rho, "= 0.5"))),
       col=c("purple","orange","palegreen4"), lty =2 , cex=0.8, bty = "n")


plot(theoretical_rho01[,1],theoretical_rho01[,2], col= "#c44601", type = "l", ylim=c(0,0.55),xlim =c(0,3),lwd=0.8,xlab =expression(gamma),ylab="Error Rate", main = "Compound Symmetric Covariance Matrix")
lines(theoretical_rho03[,1],theoretical_rho03[,2],col="#5ba300",lwd=0.8)
lines(theoretical_rho05[,1],theoretical_rho05[,2],col="#054fb9",lwd=0.8)
#abline( = 1, col = "black")
points(theoretical_rho01[,1],theoretical_rho01[,2], pch=17, col = "#c44601")
points(theoretical_rho03[,1],theoretical_rho03[,2],pch=15,col="#5ba300")
points(theoretical_rho05[,1],theoretical_rho05[,2], pch=19,col="#054fb9")
legend(x = 2, y = 0.5, cex =1.2, legend = expression(paste(cov(x[i], x[j]), " = 0.1"), paste(cov(x[i], x[j]), " = 0.3"), paste(cov(x[i], x[j])," = 0.5")), 
       col = c("#c44601", "#5ba300", "#054fb9"), 
       lty = 1, lwd = c(1, 1, 1), 
       pch = c(17, 15, 19),
       bty = "n")
theoretical_rho01 <- theoretical_rho01[apply(theoretical_rho01, 1, function(row) !all(row == 0)), ]
theoretical_rho03 <- theoretical_rho03[apply(theoretical_rho03, 1, function(row) !all(row == 0)), ]
theoretical_rho05 <- theoretical_rho05[apply(theoretical_rho05, 1, function(row) !all(row == 0)), ]


######COMPOUND SYMMEtrIC NEGAIVE#################################################
groupmean1 = 0 #0 - delta/2
groupmean2 = 1 #delta/2
gamma_seq1 <- seq(0.01, 1.5, length=10) 
gamma_seq2 <- seq(1.55, 4, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

error_rates <- NULL;gamma1 <- NULL
#foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar% {
for ( i in 1:2) {
  p <- p_seq[i]
  n <- nk_seq[i]
  err <- rep(0, 100)
  for (ind in 1:100) {
    mu1 <- rep(groupmean1,p) ; mu2 <- rep(groupmean2,p);
    #sigma <- diag(p);
   # sigma <- generate_cov_matrix(p)
    sigma <- diag(p); sigma[outer(1:p, 1:p, function(i,j) i!=j)] <- -0.3; sigma <- nearPD(sigma)$mat
    group1 <- mvrnorm(n,mu1,sigma)
    group2 <- mvrnorm(n,mu2,sigma)
    xbar1 <- colMeans(group1)
    xbar2 <- colMeans(group2)
    s_inv <-  ginv((cov(group1) + cov(group2))/2)
    err[ind] <- lda_error_rate(mu1, mu2, as.matrix(sigma), xbar1, xbar2, s_inv)
  } 
  error_rates[i] <- mean(err)
  gamma1[i] <- p / (n*2)
  #return(data.frame(error_rates = error_rates, gamma = gamma1))
}#-> results

theory_neg05 <- cbind(gamma1, error_rates)
theory_neg07 <- cbind(gamma1, error_rates)
theory_neg03 <- cbind(gamma1, error_rates)

plot(gamma1, error_rates)

plot(theory_neg05[1:20,1], theory_neg05[1:20,2], ylim = c(0.3,0.55), main = "Compound Symmetric Covariance Matrix (Theoretical)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(theory_neg05[1:20,1], theory_neg05[1:20,2], lwd = 1.2, col = "darkred")
#points(results_sparse_50[1:20,3], results_sparse_50[1:20,1], pch = 17, col = "darkgreen")
#lines(results_sparse_50[1:20,3], results_sparse_50[1:20,1], lwd =2, col = "darkgreen")
points(theory_neg07[1:20,1], theory_neg07[1:20,2], pch = 25, col = "darkblue")
lines(theory_neg07[1:20,1], theory_neg07[1:20,2], lwd =1.2, col = "darkblue")
points(theory_neg03[1:20,1], theory_neg03[1:20,2], pch = 15, col = "orange")
lines(theory_neg03[1:20,1], theory_neg03[1:20,2], lwd = 1.2, col = "orange")
legend(x =3, y = 0.45,
       legend = expression(paste(rho, " = -0.3"), paste(rho, " = -0.5"), paste(rho, " = -0.7")), 
       col = c("orange","darkred","darkblue"), bty = "n",
       lty = 1, 
       pch = c(15,19,25),
       cex = 1, 
       text.col = "black")



######AR1covmatrix############################################################

gamma_seq1 <- seq(0.01, 1, length=10) 
gamma_seq2 <- seq(1.02, 3, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq

rho = 0.3
rho = 0.5
rho = 0.7

gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)
#for (i in 1:20) {
foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar% {
  n <- nk_seq[i]
  p <- p_seq[i]
  err <- rep(0, 20)
  # Different means
  mu1 <- rep(1, p)
  mu2 <- rep(0, p)
  # Common covariance matrix- Identity 
  sigma <- diag(p); sigma = rho^abs(outer(1:p, 1:p, "-"))
  for ( s in 1: 16) {
    #sample 50 times and average the LDA error rates 
    group1 <- mvrnorm(n,mu1,sigma)
    group2 <- mvrnorm(n,mu2,sigma)
    xbar1 <- colMeans(group1)
    xbar2 <- colMeans(group2)
    s_inv <-  ginv((cov(group1) + cov(group2))/2)
    #s_inv <- solve(cov(rbind(group1, group2)))
    # Compute error rate
    err[s] <- lda_error_rate(mu1, mu2, sigma, xbar1, xbar2, s_inv)
  } 
  error_rates <- mean(err)
  gamma1 <- p / (2*n)
  return(data.frame(gamma = gamma1, result = error_rates ))
} -> results

plot(results[,1],results[,2])
ar1_rho03  =  results
  #cbind(gamma1, error_rates)
ar1_rho05  =  results
  #cbind(gamma1, error_rates)
ar1_rho07  =  results
  #cbind(gamma1, error_rates)
df_the_ar1 <- cbind(ar1_rho03, ar1_rho05, ar1_rho07)
write.csv(df_the_ar1, "theoretical_ar1.csv", row.names = FALSE)


plot(gamma1[-11], error_rates[-11])
plot(gamma1, error_rates)


plot(ar1_rho03[,1],ar1_rho03[,2], col= "#8e0bca", type = "l" , ylim=c(0,0.4),xlim =c(0,3),lwd=1.2,xlab =expression(gamma),ylab="Error Rate", main = "AR(1) Covariance Matrix")
lines(ar1_rho03[,1],ar1_rho05[,2],col="#e1111e",lwd=1.2)
lines(ar1_rho07[,1],ar1_rho07[,2],col="#08b128",lwd=1.2)
#abline( = 1, col = "black")
points(ar1_rho03[,1],ar1_rho03[,2], pch=17, col = "#8e0bca")
points(ar1_rho03[,1],ar1_rho05[,2],pch=15,col="#e1111e")
points(ar1_rho07[,1],ar1_rho07[,2], pch=19,col="#08b128")
legend(x = 2, y = 0.4, cex =1, legend = expression(paste(rho, " = 0.3"), paste(rho, " = 0.5"), paste(rho, " = 0.7")), 
       col = c("#8e0bca", "#e1111e", "#08b128"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(17, 15, 19),
       bty = "n")

####### RANDOM ENTRIES COVARIANCE MATRIX ###########################################################################################

delta = 2
groupmean1 = 0 #0 - delta/2
groupmean2 = 1 #delta/2
gamma_seq1 <- seq(0.01, 1.5, length=10) 
gamma_seq2 <- seq(1.55, 4, length=10)
gamma_seq <- c(gamma_seq1,gamma_seq2)
gamma_seq
nk_seq <- floor(seq(20, 50, length=20)) #training sample size, each group
nk_seq
p_seq <- ceiling(nk_seq*2 * gamma_seq)  #training sample size * gamma
p_seq


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

generate_cov_matrix <- function(dim) {
  # Step 1: Create a matrix with random off-diagonal entries from -1 to 1
  mat <- matrix(runif(dim*dim, min = -1, max = 0), ncol = dim)
  
  # Step 2: Set the diagonal entries to 1
  diag(mat) <- 2
  
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
set.seed(1); SIGMA <- generate_cov_matrix(400)
error_rates <- NULL;gamma1 <- NULL
#foreach(i = 1:20, .packages = c("caret","MASS","mvtnorm","Matrix"), .combine = 'rbind') %dopar% {
for ( i in 1:20) {
    p <- p_seq[i]
    n <- nk_seq[i]
    err <- rep(0, 100)
    for (ind in 1:100) {
      mu1 <- rep(groupmean1,p) ; mu2 <- rep(groupmean2,p);
      #sigma <- diag(p);
      #sigma <- generate_cov_matrix(p)
      #sigma <- rWishart(1, n*2, diag(p))[,,1]
      sigma <- SIGMA[1:p, 1:p]
        group1 <- mvrnorm(n,mu1,sigma)
        group2 <- mvrnorm(n,mu2,sigma)
        xbar1 <- colMeans(group1)
        xbar2 <- colMeans(group2)
        s_inv <-  ginv((cov(group1) + cov(group2))/2)
        err[ind] <- lda_error_rate(mu1, mu2, as.matrix(sigma), xbar1, xbar2, s_inv)
    } 
      error_rates[i] <- mean(err)
      gamma1[i] <- p / (n*2)
    # return(data.frame(error_rates = error_rates, gamma = gamma1))
    } #-> results

plot(gamma1, error_rates)
gamma1[i] <- p/(n*2)
ind
error_rates[i] <- mean(err[1:(ind-1)])

plot(results[,2],results[,1])

theory_01 <- cbind(gamma1, error_rates)
theory_neg10 <- cbind(gamma1, error_rates)
theory_neg11 <- cbind(gamma1, error_rates) #results
theory_neg0505 <- cbind(gamma1, error_rates)
theory_neg0303 <- cbind(gamma1, error_rates)
write.csv(theory_neg11, "theory_neg11NEW.csv", row.names = FALSE)
write.csv(theory_01, "theory_01NEW.csv", row.names = FALSE)
write.csv(theory_neg10, "theory_01NEW.csv", row.names = FALSE)
write.csv(theory_neg0505, "theory_neg0505.csv", row.names = FALSE)
write.csv(theory_neg0303, "theory_neg0303.csv", row.names = FALSE)

plot(theory_neg10[1:20,1], theory_neg10[1:20,2], ylim = c(0,0.55), main = "Random Entries Covariance Matrix (Theoretical)", xlab = expression(gamma), ylab = "Error Rate", pch = 19, col = "darkred")
lines(theory_neg10[1:20,1], theory_neg10[1:20,2], lwd = 1.2, col = "darkred")
#points(theory_neg11[1:20,1], theory_neg11[1:20,2], pch = 25, col = "darkblue")
#lines(theory_neg11[1:20,1], theory_neg11[1:20,2], lwd =1.2, col = "darkblue")
points(theory_01[1:20,1], theory_01[1:20,2], pch = 25, col = "darkblue")
lines(theory_01[1:20,1], theory_01[1:20,2], lwd =1.2, col = "darkblue")
points(theory_neg11[1:20,1], theory_neg11[1:20,2], pch = 17, col = "darkgreen")
lines(theory_neg11[1:20,1], theory_neg11[1:20,2], lwd =1.2, col = "darkgreen")
legend(x =2.5, y = 0.55,
       legend = expression(paste("unif(-1,0)"), paste("unif(0,1)"), paste("unif(-1,-1)")), 
       col = c("darkred","darkblue","darkgreen", "orange"), bty = "n",
       lty = 1, 
       pch = c(19,25,17, 15),
       cex = 1, 
       text.col = "black")


###############Wang Jiang theorem 2.3 ###################################################################################################
wang_error <- function(maha, n1, n2, p) {
  y1 <- p/n1 
  y2 <- p/n2
  y  <- p/(n1 + n2)
  group1_value <- (maha - (y1 - y2)* sqrt(1 - y))/(2*sqrt(maha + y1 + y2))   
  group2_value <- (maha + (y1 - y2)* sqrt(1 - y))/(2*sqrt(maha + y1 + y2))
  rate <- 0.5*pnorm(-group1_value) + 0.5*pnorm(-group2_value)
  return(rate)
}

wang_error2 <- function(maha, y1, y2, y) {
  group1_value <- (maha - (y1 - y2)* sqrt(1 - y))/(2*sqrt(maha + y1 + y2))   
  group2_value <- (maha + (y1 - y2)* sqrt(1 - y))/(2*sqrt(maha + y1 + y2))
  rate <- 0.5*pnorm(-group1_value) + 0.5*pnorm(-group2_value)
  return(rate)
}

wang_error3 <- function(maha,  y) {
 # group1_value <- (maha)*(sqrt(1-y)) /(2*sqrt(maha + 4*y))   
  group2_value <- (maha) /(2*sqrt(maha + 4*y))
  rate <- pnorm(-(sqrt(1-y))*group2_value)
  return(rate)
}


theorem1 <- function(maha, gamma) {
  val <- 2*sqrt(maha)*sqrt(gamma*(1-gamma))
  return(pnorm(-val))
}

theorem1(5, 0)
theorem1(100, 1)
#theorem1(1, 0.5)

theorem1(1, 0.5)

theorem1(100, 0.9)
xval <- seq(0,1,by = 0.1)
plot(xval, theorem1(100,xval ))

plot(xval, theorem1(5,xval ))
plot(xval, theorem1(1,xval ))
#delta  = 1 ,sigma = I 

maha_distance <- function(delta, sigma){
  sig_inverse <- ginv(sigma) 
  distance <- t(delta)%*%sig_inverse %*% delta
  return(distance)
}

nk_seq <- floor(seq(50, 200, length=20)) #training sample size, each group
nk_seq
#total dataset training + testing 
gamma_seq <- seq(0.001, 1, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq * gamma_seq)  #training sample size * gamma
p_seq
gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)
for (i in 1:20) {
  n <- nk_seq[i]
  p <- p_seq[i]
  err <- rep(0, 20)
  # Different means
  delta = rep(1,p)
  # Common covariance matrix- Identity 
  sigma <- diag(p);   
  #sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.3
  #sigma = rho^abs(outer(1:p, 1:p, "-"))
  maha = as.numeric(maha_distance(delta,sigma))
  error_rates[i] <- wang_error3(maha, p/n)
  gamma1[i] <- p / n
}
identity_wang05 <- cbind(gamma1, error_rates)
identity_wang1 <- cbind(gamma1, error_rates)
identity_wang2 <- cbind(gamma1, error_rates)
df_wang1 <- cbind(identity_wang05, identity_wang1, identity_wang2)
write.csv(df_wang1, "wang_identity_delta.csv", row.names = FALSE)


plot(identity_wang05[,1],identity_wang05[,2] , xlab = expression(gamma), ylab = "Error Rate", type = "l", ylim=c(0,0.5), lwd =1.2, main = "Identity Covariance Matrix")
lines(identity_wang1[,1],identity_wang1[,2], col = "darkblue",lwd=1.2 )
lines(identity_wang2[,1],identity_wang2[,2], col = "darkred", lwd=1.2)
points(identity_wang05[,1],identity_wang05[,2], pch = 16)
points(identity_wang1[,1],identity_wang1[,2], col = "darkblue", pch = 17)
points(identity_wang2[,1],identity_wang2[,2], col = "darkred", pch = 18)
legend(x = 0.4, y = 0.5, cex =1, legend = expression(paste(delta, " = 0.5"), paste(delta, " = 1"), paste(delta, " = 2")), 
       col = c("black", "darkblue", "darkred"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(16, 17, 18),
       bty = "n")

################ Mahalanobis distance##################

maha_distance <- function(delta, sigma, n){
  sig_inverse <- ginv(sigma) 
  distance <- t(delta)%*%sig_inverse %*% delta
  return(distance/n)
}
mahalanobis_id <-rep(0, 100)
mahalanobis_cs <- rep(0, 100)
mahalanobis_rand <- rep(0, 100)
maha_delta3 <- rep(0,100)
sigma_wish <- rWishart(1, 100, diag(100))[,,1]*(1/50)

for (i in 1:100) {
  p <- i 
  mu1 <- rep(0,p)  
  mu2 <- rep(1,p)  
  sigma <- diag(p) 
  #diag(sigma) <- diag(sigma) +5
  #sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.1
  #sigma = 0.3^abs(outer(1:p, 1:p, "-"))
  #sigma =sigma_wish[1:p, 1:p]
  #delta = mu1 - mu2
  #mahalanobis_id[i] <- maha_distance(mu1-mu2, sigma, 100)
  #mahalanobis_cs[i] <- maha_distance(mu1-mu2, sigma, 100)
   #mahalanobis_rand[i] <- maha_distance(mu1-mu2, sigma, 100)
  maha_delta3[i] <- maha_distance(mu1-mu2, sigma,100)
  
}
maha_de

plot(seq(0,1, length = 100), maha_delta3)
plot(seq(0,1, length = 100), maha_delta3, xlab = expression(p),main = "Scaled Mahalanobis Distance vs. Dimension", pch = 19,ylab = expression(paste("Squared Mahalanobis Distance ", Delta^2)), col = "brown")
lines(seq(0,1, length = 100), maha_delta3, col = "brown")


plot(seq(0,1, length = 100), mahalanobis_id, xlab = expression(gamma),main = expression(paste("Scaled Mahalanobis Distance against ", gamma)), pch = 19, ylab = expression(Delta["*"]), col = "brown")
lines(seq(0,1, length = 100), mahalanobis_id, col = "brown")
points(seq(0,1, length = 100), mahalanobis_cs, pch = 19, col = "darkgreen")
lines(seq(0,1, length = 100), mahalanobis_cs, col = "darkgreen")
points(seq(0,1, length = 100), mahalanobis_rand,col = "purple" )
lines(seq(0,1, length = 100), mahalanobis_rand,col = "purple")
legend(x = 0.05, y =0.95, cex =1, legend = c("Identity", "Compound Symmetric ", "AR(1)"), title = "Covariance Matrix Structure",
       col = c("brown", "darkgreen", "purple"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       #pch = c(19, 19, 19),
       bty = "n")


#########COMPOUND SYMMETRIC########################################


nk_seq <- floor(seq(50, 400, length=20)) #training sample size, each group

#total dataset training + testing 
gamma_seq <- seq(0.001, 1, length=20) 
gamma_seq
p_seq <- ceiling(nk_seq * gamma_seq)  #training sample size * gamma
p_seq
rho = 0.1
rho = 0.3  #http://127.0.0.1:42077/graphics/plot_zoom_png?width=723&height=706
rho = 0.5 
gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)


theorem1 <- function(maha, gamma) {
  val <- 2*sqrt(maha)*sqrt(gamma*(1-gamma))
  return(pnorm(-val))
}
maha_distance <- function(delta, sigma, n){
  sig_inverse <- ginv(sigma) 
  distance <- t(delta)%*%sig_inverse %*% delta
  return(distance/n)
}
gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)

for (i in 1:20) {
  n <- nk_seq[i]
  p <- p_seq[i]
  err <- rep(0, 20)
  # Different means
  delta = rep(3,p)
  # Common covariance matrix- Identity 
  sigma <- diag(p);   
  sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.5  #; sigma <- as.matrix(nearPD(sigma)$mat)
  #sigma = rho^abs(outer(1:p, 1:p, "-"))
  maha = as.numeric(maha_distance(delta,sigma, n))
  error_rates[i] <- theorem1(maha,p / n )    #maha    #wang_error3(maha, p/n)
  gamma1[i] <- p / n
}

plot(gamma1,error_rates)


cs_wang03 <- cbind(gamma1, error_rates)
cs_wang01 <- cbind(gamma1, error_rates)
cs_wang05 <- cbind(gamma1, error_rates)

df_wang2 <- read.csv("wang_compoundsymmetric.csv", header = T)
df_wang2 <- cbind(cs_wang01, cs_wang03, cs_wang05)
write.csv(df_wang2, "wang_compoundsymmetric.csv", row.names = FALSE)


plot(df_wang2[,1],df_wang2[,2] , xlab = expression(gamma), ylab = "Error Rate", type = "l", ylim=c(0,0.5), lwd =1.2, main = "Identity Covariance Matrix")
lines(df_wang2[,1],df_wang2[,4], col = "darkblue",lwd=1.2 )
lines(df_wang2[,1],df_wang2[,6], col = "darkred", lwd=1.2)
points(df_wang2[,1],df_wang2[,2], pch = 16)
points(df_wang2[,1],df_wang2[,4], col = "darkblue", pch = 17)
points(df_wang2[,1],df_wang2[,6], col = "darkred", pch = 18)
legend(x = 0.5, y = 0.15, cex =1, legend = expression(paste(delta[p], " = 0.1"), paste(delta[p], " = 0.5"), paste(delta[p], " = 3")), 
       col = c("darkred", "black", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(18, 16, 17),
       bty = "n")


plot(df_wang2[,1],df_wang2[,2] , xlab = expression(gamma), ylab = "Error Rate", type = "l", ylim=c(0,0.5), lwd =1.2, main = "Identity Covariance Matrix")
lines(df_wang2[,1],df_wang2[,4], col = "darkblue",lwd=1.2 )
lines(df_wang2[,1],df_wang2[,6], col = "darkred", lwd=1.2)
points(df_wang2[,1],df_wang2[,2], pch = 16)
points(df_wang2[,1],df_wang2[,4], col = "darkblue", pch = 17)
points(df_wang2[,1],df_wang2[,6], col = "darkred", pch = 18)
legend(x = 0.5, y = 0.15, cex =1, legend = expression(paste(cov(x[i], x[j]), " = 0.1"), paste(cov(x[i], x[j]), " = 0.3"), paste(cov(x[i], x[j]), " = 0.5")), 
       col = c("darkred", "black", "darkblue"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(18, 16, 17),
       bty = "n")



cs_wang03 <- cbind(gamma1, error_rates)
cs_wang01 <- cbind(gamma1, error_rates)
cs_wang05 <- cbind(gamma1, error_rates)

df_wang2 <- read.csv("wang_compoundsymmetric.csv", header = T)
df_wang2 <- cbind(cs_wang01, cs_wang03, cs_wang05)
write.csv(df_wang2, "wang_compoundsymmetric.csv", row.names = FALSE)


plot(df_wang2[,1],df_wang2[,2] , xlab = expression(gamma), ylab = "Error Rate", type = "l", ylim=c(0,0.5), lwd =1.2, main = "Compound Symmetric Covariance Matrix")
lines(df_wang2[,1],df_wang2[,4], col = "darkblue",lwd=1.2 )
lines(df_wang2[,1],df_wang2[,6], col = "darkred", lwd=1.2)
points(df_wang2[,1],df_wang2[,2], pch = 16)
points(df_wang2[,1],df_wang2[,4], col = "darkblue", pch = 17)
points(df_wang2[,1],df_wang2[,6], col = "darkred", pch = 18)
legend(x = 0.6, y = 0.1, cex =1, legend = expression(paste(cov(X[i], X[j]), " = 0.1"), paste(cov(X[i], X[j]), " = 0.3"), paste(cov(X[i], X[j]), " = 0.5")), 
       col = c("black", "darkblue", "darkred"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(16, 17, 18),
       bty = "n")


#####################AR1 COV STRUCTURE########################################

rho = 0.1
rho = 0.3  #http://127.0.0.1:42077/graphics/plot_zoom_png?width=723&height=706
rho = 0.5 
gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)
for (i in 1:20) {
  n <- nk_seq[i]
  p <- p_seq[i]
  err <- rep(0, 20)
  # Different means
  delta = rep(1,p)
  # Common covariance matrix- Identity 
  sigma <- diag(p);   
  #sigma[lower.tri(sigma) | upper.tri(sigma)] <- 0.5
  sigma = rho^abs(outer(1:p, 1:p, "-"))
  maha = as.numeric(maha_distance(delta,sigma))
  error_rates[i] <- wang_error3(maha, p/n)
  gamma1[i] <- p / n
}

plot(gamma1,error_rates)

ar1_wang03 <- cbind(gamma1, error_rates)
ar1_wang01 <- cbind(gamma1, error_rates)
ar1_wang05 <- cbind(gamma1, error_rates)
df_wang3 <- cbind(ar1_wang01, ar1_wang03, ar1_wang05)
write.csv(df_wang3, "wang_ar1.csv", row.names = FALSE)
df_wang3 <- read.csv("wang_ar1.csv", header = T)


plot(df_wang3[,1],df_wang3[,2] , xlab = expression(gamma), ylab = "Error Rate", type = "l", ylim=c(0,0.5), lwd =1.2, main = "AR(1) Covariance Matrix")
lines(df_wang3[,1],df_wang3[,4], col = "darkblue",lwd=1.2 )
lines(df_wang3[,1],df_wang3[,6], col = "darkred", lwd=1.2)
points(df_wang3[,1],df_wang3[,2], pch = 16)
points(df_wang3[,1],df_wang3[,4], col = "darkblue", pch = 17)
points(df_wang3[,1],df_wang3[,6], col = "darkred", pch = 18)
legend(x = 0.4, y = 0.5, cex =1, legend = expression(paste(cov(x[i], x[j]), " = 0.1"), paste(cov(x[i], x[j]), " = 0.3"), paste(cov(x[i], x[j]), " = 0.5")), 
       col = c("black", "darkblue", "darkred"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(16, 17, 18),
       bty = "n")

########RANDOM COVARIANCE MATRIX#####################################################


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

generate_cov_matrix <- function(dim) {
  # Step 1: Create a matrix with random off-diagonal entries from -1 to 1
  mat <- matrix(runif(dim*dim, min = -0.3, max = 0.3), ncol = dim)
  diag(mat) <- 1
  mat <- (mat + t(mat))/2

  mat <- nearPD(mat)$mat
  return(mat)
}
gamma1 <- numeric(length(nk_seq));error_rates <- rep(0, 20)
for (i in 1:20) {
  n <- nk_seq[i]
  p <- p_seq[i]
  err <- rep(0, 100)
  # Different means
  delta = rep(1,p)
  #sigma <- diag(p);   
  for ( ind in 1:200 ) {
 #sigma <- generate_cov_matrix(p)
    sigma <- rWishart(1, n, diag(p))[,,1]
  maha = as.numeric(maha_distance(delta,as.matrix(sigma)))
  err[ind] <- wang_error3(maha, p/n)
  }
  error_rates[i] <- mean(err)
  gamma1[i] <- p / n
}
wang_neg0303 <- cbind(gamma1, error_rates)
wang_neg0505 <- cbind(gamma1, error_rates)
wang_neg11 <- cbind(gamma1, error_rates)
wang_01 <- cbind(gamma1, error_rates)
plot(gamma1,error_rates)


plot(wang_neg0303[,1],wang_neg0303[,2] , xlab = expression(gamma), ylab = "Error Rate", type = "l", ylim=c(0,0.5), lwd =1.2, main = "Random Entries Matrix")
lines(wang_neg0303[,1],wang_neg0303[,2], col = "darkblue",lwd=1.2 )
lines(wang_neg0505[,1],wang_neg0505[,2], col = "darkred", lwd=1.2)
points(wang_neg0505[,1],wang_neg0505[,2], pch = 16)
points(wang_neg11[,1],wang_neg11[,2], col = "darkblue", pch = 17)
points(wang_neg11[,1],wang_neg11[,2], col = "darkred", pch = 18)
legend(x = 0.4, y = 0.5, cex =1, legend = expression(paste(rho, " = 0.1"), paste(rho, " = 0.3"), paste(rho, " = 0.5")), 
       col = c("black", "darkblue", "darkred"), 
       lty = 1, lwd = c(1.2, 1.2, 1.2), 
       pch = c(16, 17, 18),
       bty = "n")


p <- 3  # Dimension of the covariance matrix
n <- 4  # Degrees of freedom (should be > p - 1 for the matrix to be invertible)
Sigma <- diag(p)  # Scale matrix (for simplicity, we use the identity matrix here)

# Generate a matrix from the Wishart distribution
set.seed(123)
A <- rWishart(1, n, Sigma)[,,1]




