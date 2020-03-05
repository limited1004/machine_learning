# Метрические алгоритмы классификации
### kNN
 Алгоритм k ближайших соседей - kNN относит объект u к тому классу элементов которого больше среди k ближайших соседей 
![equation](http://latex.codecogs.com/gif.latex?x_u^{i},&space;i=1,...,k:)

![equation](http://latex.codecogs.com/gif.latex?w(i,&space;u)&space;=&space;[i&space;\leq&space;k];&space;a(u;&space;X^l,&space;k)&space;=&space;argmax_{y\epsilon&space;Y}&space;\sum^k_{i&space;=&space;1}{[y^i_{u}&space;=&space;y]})

Для выбора оптимального k используют метод скользащего контроля (LOO).
Применив kNN и LOO к датасету Ириса Фишера получим результат:

![](https://github.com/limited1004/machine_learning/blob/master/imgs/LOO_kNN.png)


Видно, что лучишй результат получаем при k = 6, с оценкой ошибки равной 0.33, что равно 96% успешных классификаций.

 kNN — один из простейших алгоритмов классификации, поэтому на реальных задачах он зачастую оказывается неэффективным. Помимо точности классификации, проблемой этого классификатора является скорость классификации: если в обучающей выборке N объектов, в тестовой выборе M объектов, а размерность пространства — K, то количество операций для классификации тестовой выборки может быть оценено как O(K*M*N).

Карта классификации для kNN:  
![](https://github.com/limited1004/machine_learning/blob/master/imgs/map_knn.png)
### kwNN
Алогоритм k взвешенных ближайших соседей:
![equation](http://latex.codecogs.com/gif.latex?w(i,&space;u)&space;=&space;[i&space;\leq&space;k]w(i);&space;a(u;&space;X^l,&space;k)&space;=&space;argmax_{y\epsilon&space;Y}&space;\sum^k_{i&space;=&space;1}{[y^i_{u}&space;=&space;y]}w(i))
возьмем за вес ![equation](http://latex.codecogs.com/gif.latex?w(i)&space;=&space;q^i,q\epsilon&space;(0,1)), и его же будем перебирать по LOO при фиксированном k = 6, получим результат:
![](https://github.com/limited1004/machine_learning/blob/master/imgs/loo_kwnn.png)
Видем что лучший результат при k = 6 и q = 1. Равен 0.33, что примерно 96% успешных классификаций.
Зачем использовать kwNN если там больше расчетов? В задачах с числом классов 3 и более нечётность уже не помогает и сутации неодназначности могут возниктаь. Тогда на помошь приходят веса, и объект классифицируется к тому классу, чей суммарны вес больше среди k соседий.


Карта классификации для kWNN:  
![](https://github.com/limited1004/machine_learning/blob/master/imgs/map_kWNN.png)


# Байесовские алгоритмы классификации
### Линии уровня 
1) Если признаки некоррелированы, ![equation](http://latex.codecogs.com/gif.latex?\sum&space;=&space;diag(\sigma_1^2,...,\sigma_1^2)), то линии уровня плотности распределения имеют форму эллипсоидов с центром ![equation](http://latex.codecogs.com/gif.latex?\mu))

2) Если признаки имеют одинаковые дисперсии, ![equation](http://latex.codecogs.com/gif.latex?\sum&space;=&space;\sigma^2I_n)

3) Если признаки коррелированы, то матрица ![equation](http://latex.codecogs.com/gif.latex?\sum) не диагональна и линии уровня имеют форму эллипсоидов, оси которых повернуты ( направлены вдоль собственных векторов матрицы ![equation](http://latex.codecogs.com/gif.latex?\sum) ) относительно исходной системы координат.

С помощью ортогонального преобразования ![equation](http://latex.codecogs.com/gif.latex?x'=V^Tx,&space;V&space;=&space;(v_1,...,v_n)). - ортогональные собственные векторы матрицы ![equation](http://latex.codecogs.com/gif.latex?\sum), можно перейти к первому случаю.

Результат реализации линий уровня для нормального распределения: 
![](https://github.com/limited1004/machine_learning/blob/master/imgs/lines_result.png)


# Линейные классификаторы
### Adaline
  Алгоритм классификации ADALINE— адаптивны линейный элемент, в качестве функции потерь используется квадратичная функция потерь:
![equation](http://latex.codecogs.com/gif.latex?(<w,x>&space;-&space;y_i)^2)
 Алгоритм обучается с помошью стохастического градиента.

  Обучение ADALINE заключается в подборе "наилучших" значений вектора весов w. Какие значение весов лучше определяет функционал потерь. В ADALINE используется функционал, предложенный Видроу и Хоффом, ![equation](http://latex.codecogs.com/gif.latex?L(a,x)&space;=&space;(a-y)^2). Таким образом необходимо минимизировать функционал ![equation](http://latex.codecogs.com/gif.latex?L(a,x)&space;=&space;Q(w)): 

![equation](http://latex.codecogs.com/gif.latex?$$Q(w)=&space;\sum^m_{i=1}{(a(x_i,w)-y_i)^2}\rightarrow&space;min_w$$).
Построим алгоритм на классификации ирисов Фишера, с классами virginica и versicolo по 3 и 4 параметрам datafram'а. Проверим 
работу алгоритма на 2,5,10 и 43 шагах стохастического градиента. Получим: 
![](https://github.com/limited1004/machine_learning/blob/master/imgs/adadline_result.png)

Линия на графике - разделяющая гиперплоскость. Оптимум достигается на 43'ом шаге.
Веса получились равны: w_1 = 3.24, w_2 = -1.7

#### Перцептрон Розенблата
Будем ипсользовать virginica и versicolor. По 3 и 4 параметрам datafram'а

Будем проверять результаты работы на 10, 75, 230, 275 шаге стохастического градиентного спуска. В реузльтате получим:

![](https://github.com/limited1004/machine_learning/blob/master/imgs/perceptron.png)

Линия на графике - раздиляющая гиперплоскость. Оптимум достигается при 275'ом шаге.
В результате получили веса, равные w_1 = -102.9, w_2 = 152.1, w_3 = 119.
