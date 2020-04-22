## ������������ ��������� �������
trainingSampleNormalization <- function(xl)
{
  n <- dim(xl)[2] - 1
  for(i in 1:n)
  {
    xl[, i] <- (xl[, i] - mean(xl[, i])) / sd(xl[, i])
  }
  return (xl)
}
## ���������� ������� ��� �� -1 ��� w0
trainingSamplePrepare <- function(xl)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  xl <- cbind(xl[, 1:n], seq(from = -1, to = -1,
                             length.out = l), xl[, n + 1])
}
  ## ������������ ������� ������
  lossQuad <- function(x)
  {
    return ((x-1)^2)
  }
  ## �������������� �������� ��� ADALINE
  sg <- function(xl,coll, eta = 1, lambda = 1/6, eps = 1e-5) {
    l <- dim(xl)[1]
    n <- dim(xl)[2] - 1
    w <- rep(0.5, n)
    iterCount <- 0
    
    ## initialize Q
    Q <- 0
    
    for (i in 1:l)
    {
      ## calculate the scalar product <w,x>
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      mi <-sum(w*xi) * yi
      Q <- Q + (mi - 1)^2
    }
    
    iterCount <- 0
    repeat
    {
      if (iterCount > 1000) {
        
        break
      }
      
      ## calculate the margins for all objects of the training sample
      margins <- array(dim = l)
      for (i in 1:l){
        xi <- xl[i, 1:n]
        yi <- xl[i, n + 1]
        
        margins[i] <- crossprod(w , xi) * yi
        
      }
      
      ## select the error objects
      errorIndexes <- which(margins <= 0)
      if (length(errorIndexes) > 0)
      {
        # select the random index from the errors
        i <- sample(errorIndexes, 1)
        iterCount <- iterCount + 1
        xi <- xl[i, 1:n]
        yi <- xl[i, n + 1]
        
        
        ## calculate an error
        mi <- sum(w*xi)*yi
        ex <- (mi-1)^2
        ld <- (mi/yi - yi) * xi
        eta <- 1/iterCount
        w <- w - eta * ld
        x <- seq(-2, 2, len = 100)
        f <- function(x) {
          return( - x*w[1]/w[2] + w[3]/w[2] )
        }
        y <- f(x)
        lines(x, y, type="l",col=coll)
        
        ## Calculate a new Q
        Qprev <- Q
        Q <- (1 - lambda) * Q + lambda * ex
        if (abs(Q - Qprev) < 1e-5) {
          break
        }
        
      }
      else
      {
        break
      }
      
    }
    return(w)
  }
## ������� ������ ��� ������� �����
lossPerceptron <- function(x)
{
  return (max(-x, 0))
}
## �������������� �������� � �������� �����
sg.Hebb <- function(xl, eta = 0.1, lambda = 1/6)
{
  x = 1
  abline(a = w[3] / w[2], b = -w[1] / w[2], lwd = x, col =
           "blue")
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1/2, 1/2, 1/2)
  iterCount <- 0
  
  ## initialize Q
  Q <- 0
  for (i in 1:l)
  {
    
    ## calculate the scalar product <w,x>
    wx <- sum(w * xl[i, 1:n])
    ## calculate a margin
    margin <- wx * xl[i, n + 1]
    # Q <- Q + lossQuad(margin)
    Q <- Q + lossPerceptron(margin)
  }
  iterCount <- 0
  repeat
  {
    if (iterCount > 1000) {
      
      break
    }
        ## ����� ��������� ��������
    margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      
      margins[i] <- crossprod(w, xi) * yi
    }
    
    ## ������� ��������� �������
    errorIndexes <- which(margins <= 0)
    if (length(errorIndexes) > 0)
    {
      # ������� ��������� ��������� ������
      i <- sample(errorIndexes, 1)
      
      iterCount <- iterCount + 1
      
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      
      w <- w + eta * yi * xi
      x <- seq(-2, 2, len = 100)
      f <- function(x) {
        return( - x*w[1]/w[2] + w[3]/w[2] )
      }
      y <- f(x)
      lines(x, y, type="l",col='green')
    }
    else
      break;
  }
  
  return (w)
}
## ��������������� ������� ������
lossLog <- function(x)
{
  return (log2(1 + exp(-x)))
}
## ���������� �������
sigmoidFunction <- function(z)

# ���-�� �������� � ������ ������
ObjectsCountOfEachClass <- 100
## ���������� ��������� ������
library(MASS)
Sigma1 <- matrix(c(2, 0, 0, 10), 2, 2)
Sigma2 <- matrix(c(4, 1, 1, 2), 2, 2)

xy1 <- mvrnorm(n=ObjectsCountOfEachClass, c(0, 0), Sigma1)
xy2 <- mvrnorm(n=ObjectsCountOfEachClass, c(10, -10),
               Sigma2)
xl <- rbind(cbind(xy1, -1), cbind(xy2, +1))
colors <- c(rgb(255/255, 255/255, 0/255), "white",
            rgb(0/255, 200/255, 0/255))
## ������������ ������
xlNorm <- trainingSampleNormalization(xl)
xlNorm <- trainingSamplePrepare(xlNorm)
## ����������� ������
## ADALINE
plot(xlNorm[, 1], xlNorm[, 2], pch = 21, bg = colors[xl[,3]
                                                     + 2], asp = 1, xlab = "x1", ylab = "x2", main = "��������
��������������")
w <- sg(xlNorm,coll="blue")
abline(a = w[3] / w[2], b = -w[1] / w[2], lwd = 3, col = "blue")
## ������� �����
w <- sg.Hebb(xlNorm)
abline(a = w[3] / w[2], b = -w[1] / w[2], lwd = 3, col =
         "green3")

legend("bottomleft", c("ADALINE", "������� �����"), pch = c(15,15,15), col =c("blue", "green3"))