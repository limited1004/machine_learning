library(datasets)
data(iris)
iris = iris[0:100,]
colors <- c('green', 'yellow')

## ������ �������������� ������� ��� ���
estimateFisherCovarianceMatrix <- function(objects1,
                                           objects2, mu1, mu2)
{
  rows1 <- dim(objects1)[1]
  rows2 <- dim(objects2)[1]
  rows <- rows1 + rows2
  cols <- dim(objects1)[2]
  
  sigma <- matrix(0, cols, cols)

  for (i in 1:rows1)
  {
    sigma <- sigma + (t(objects1[i,] - mu1) %*%
                        (objects1[i,] - mu1)) / (rows + 2)
    print(sigma)
  }
  
  for (i in 1:rows2)
  {
    sigma <- sigma + (t(objects2[i,] - mu2) %*%
                        (objects2[i,] - mu2)) / (rows + 2)
    print(sigma)
  }
  
  return (sigma)
}

estimateMu <- function(objects)
{
  ## mu = 1 / m * sum_{i=1}^m(objects_i)
  rows <- dim(objects)[1]
  cols <- dim(objects)[2]
  
  mu <- matrix(NA, 1, cols)
  
  for (col in 1:cols)
  {
    mu[1, col] = mean(objects[,col])
  }
  
  return(mu)
}
## ���������� �������� � ������ ������
ObjectsCountOfEachClass <- 200
## ���������� ���������� MASS ��� ��������� ������������ ����������� �������������
library(MASS)
## ���������� �������� ������
Sigma1 <- matrix(c(3, 0, 0, 3), 2, 2)
Sigma2 <- matrix(c(3, 0, 0, 3), 2, 2)
Mu1 <- c(2, 10)
Mu2 <- c(8, 2)

## �������� ��� ������ � ���� �������

## ������ ��������� �������
colors <- c('green', 'yellow')
plot(iris[,1], iris[,2], pch = 21, bg = colors[iris[,5]], asp =
       1)
## ����������
objectsOfFirstClass <- iris[iris[,5] == 'setosa', 1:2]
objectsOfSecondClass <- iris[iris[,5] == 'versicolor', 1:2]
obj1 = data.matrix(objectsOfFirstClass)
obj2 = data.matrix(objectsOfSecondClass)
mu1 <- estimateMu(obj1)
mu2 <- estimateMu(obj2)
Sigma <-
  estimateFisherCovarianceMatrix(obj1,
                                 obj2, mu1, mu2)
## �������� ������������ ���
inverseSigma <- solve(Sigma)
alpha <- inverseSigma %*% t(mu1 - mu2)
mu_st <- (mu1 + mu2) / 2
beta <- mu_st %*% alpha
## ������ ���
abline(beta / alpha[2,1], -alpha[1,1]/alpha[2,1], col =
         "red", lwd = 3)