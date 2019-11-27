# Метрические алгоритмы классификации
### kNN
 Алгоритм k ближайших соседей - kNN относит объект u к тому классу элементов которого больше среди k ближайших соседей 
![equation](http://latex.codecogs.com/gif.latex?x_u^{i},&space;i=1,...,k:)

![equation](http://latex.codecogs.com/gif.latex?w(i,&space;u)&space;=&space;[i&space;\leq&space;k];&space;a(u;&space;X^l,&space;k)&space;=&space;argmax_{y\epsilon&space;Y}&space;\sum^k_{i&space;=&space;1}{[y^i_{u}&space;=&space;y]})

Для выбора оптимального k используют метод скользащего контроля (LOO).
Применив kNN и LOO к датасету Ириса Фишера получим результат:

![](https://github.com/Goncharoff/SMPR/blob/master/imgs/LOO_knn.png)


Видно, что лучишй результат получаем при k = 6, с оценкой ошибки равной 0.33, что равно 96% успешных классификаций.

 kNN — один из простейших алгоритмов классификации, поэтому на реальных задачах он зачастую оказывается неэффективным. Помимо точности классификации, проблемой этого классификатора является скорость классификации: если в обучающей выборке N объектов, в тестовой выборе M объектов, а размерность пространства — K, то количество операций для классификации тестовой выборки может быть оценено как O(K*M*N).

Карта классификации для kNN:  
![](https://github.com/Goncharoff/SMPR/blob/master/imgs/map_knn.png)
### kwNN
Алогоритм k взвешенных ближайших соседей:
![equation](http://latex.codecogs.com/gif.latex?w(i,&space;u)&space;=&space;[i&space;\leq&space;k]w(i);&space;a(u;&space;X^l,&space;k)&space;=&space;argmax_{y\epsilon&space;Y}&space;\sum^k_{i&space;=&space;1}{[y^i_{u}&space;=&space;y]}w(i))
возьмем за вес ![equation](http://latex.codecogs.com/gif.latex?w(i)&space;=&space;q^i,q\epsilon&space;(0,1)), и его же будем перебирать по LOO при фиксированном k = 6, получим результат:
![](https://github.com/Goncharoff/SMPR/blob/master/imgs/loo_kwnn.png)
Видем что лучший результат при k = 6 и q = 1. Равен 0.33, что примерно 96% успешных классификаций.
Зачем использовать kwNN если там больше расчетов? В задачах с числом классов 3 и более нечётность уже не помогает и сутации неодназначности могут возниктаь. Тогда на помошь приходят веса, и объект классифицируется к тому классу, чей суммарны вес больше среди k соседий.


Карта классификации для kWNN:  
![](https://github.com/Goncharoff/SMPR/blob/master/imgs/map_kWNN.png)
