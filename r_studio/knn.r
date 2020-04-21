library(datasets)
data(iris)

colors <- c("setosa" = "red", "versicolor" = "green3",
            "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species],
     col = colors[iris$Species])

## ��������� ����������
euclideanDistance <- function(u, v)
{
  sqrt(sum((u - v)^2))
}

## ��������� ������� �������� ���������� �� ������� z
sortObjectsByDist <- function(xl, z, metricFunction =
                                euclideanDistance)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  
  ## ������ ������� ����������
  distances <- matrix(NA, l, 2)
  
  for (i in 1:l)
  {
    distances[i, ] <- c(i, metricFunction(xl[i, 1:n], z))
  }
  
  ## ���������
  orderedXl <- xl[order(distances[, 2]), ]
  
  return (orderedXl);
}
## ��������� ����� kNN
kNN <- function(xl, z, k)
{
  ## ��������� ������� �������� �����������������
  orderedXl <- sortObjectsByDist(xl, z)
  n <- dim(orderedXl)[2] - 1
  
  ## �������� ������ ������ k �������
  classes <- orderedXl[1:k, n + 1]
  
  ## ���������� ������� ������������� ������� ������
  counts <- table(classes)
  ## ������� �����, ������� ���������� ����� ������ k �������
  class <- names(which.max(counts))
  
  return (class)
}
## ��������� ����� kwNN
kwNN <- function(xl, z, k, q) {
  orderedXl <- sortObjectsByDist(xl, z, euclideanDistance) 
  n <- dim(orderedXl)[2] - 1 
  classes <- orderedXl[1:k, n + 1] 
  counts <- table(classes)
  m <- c("setosa" = 0, "versicolor" = 0, "virginica" = 0)
  for (i in seq(1:k)){
    w <- q ^ i
    m[[classes[i]]] <- m[[classes[i]]] + w
    
  }
  
  class <- names(which.max(m)) 
  #print(m)
  
  return (class)
}

## ������ �������
colors <- c("setosa" = "red", "versicolor" = "green3",
            "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col
     = colors[iris$Species], asp = 1)
## ������������� ������ ��������� �������
q = 0.2
z <- c(2.3, 1)
xl <- iris[, 3:5]
class <- kwNN(xl, z, k=6, q)
points(z[1], z[2], pch = 22, bg = colors[class], asp = 1)
