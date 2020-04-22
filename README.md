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
### Наивный байесовский классификатор
Будем полагать, что все объекы описываются n числовыми признаками. Обозначим через ![equation](http://latex.codecogs.com/gif.latex?x=(\xi_1,...,\xi_n&space;)), произвольный элемент пространства объектов ![equation](http://latex.codecogs.com/gif.latex?X&space;=&space;\mathbb{R}^n,&space;\varepsilon&space;_i=f_i(x)). Предположим, что все признаки ![equation](http://latex.codecogs.com/gif.latex?f_1,...,f_n) являются независимыми случайными величинами. Следовательно, функции правдоподобия классов представимы в виде,![equation](http://latex.codecogs.com/gif.latex?p_y(x)=p_{y1}(\xi_1)...p_{yn}(\xi_n),&space;y\epsilon&space;Y) где ![equation](http://latex.codecogs.com/gif.latex?p_{yj}(\xi_j)) - плотность распределений значений jго признака для класса y. Оценивать n одномерных плотностей гораздо проще, чем одну n-мерную плотность. Однако данное предположение крайне редко работает на практике, поэтому алгоритмы, использующий его, называют наивным байесовким методом.
Произведем построение алогритма на модельной выборке.
Получим линейноразделимое мн-во признаков:
![](https://github.com/limited1004/machine_learning/blob/master/r_studio/naive_dots.png)

И после приминения классификатора получим карту классификации:

![](https://github.com/limited1004/machine_learning/blob/master/r_studio/naive_map.png)

Плюсы наивного байесовского классификатора:

•Простота реализации.

•Низкие вычислительные затраты при обучении и классификации.

•В тех редких случаях, когда признаки (почти) независимы, наивный байесовский классификатор (почти) оптимален.


Минусы:
• Низкое качество классификации. Он используется либо как эталон при экспериментальном сравнении алгоритмов, либо как элементарный  строительный блок при алгоритмических композициях.
# Линейные классификаторы
### Adaline
  Алгоритм классификации ADALINE— адаптивны линейный элемент, в качестве функции потерь используется квадратичная функция потерь:
![equation](http://latex.codecogs.com/gif.latex?(<w,x>&space;-&space;y_i)^2)
 Алгоритм обучается с помошью стохастического градиента.

  Обучение ADALINE заключается в подборе "наилучших" значений вектора весов w. Какие значение весов лучше определяет функционал потерь. В ADALINE используется функционал, предложенный Видроу и Хоффом, ![equation](http://latex.codecogs.com/gif.latex?L(a,x)&space;=&space;(a-y)^2). Таким образом необходимо минимизировать функционал ![equation](http://latex.codecogs.com/gif.latex?L(a,x)&space;=&space;Q(w)): 

![equation](http://latex.codecogs.com/gif.latex?$$Q(w)=&space;\sum^m_{i=1}{(a(x_i,w)-y_i)^2}\rightarrow&space;min_w$$).
Построим алгоритм на классификации ирисов Фишера, с классами virginica и versicolo по 3 и 4 параметрам datafram'а. Проверим 
работу алгоритма на 2,5,10 и 43 шагах стохастического градиента. Получим: 

Линии на графике - разделяющие гиперплоскости. 
Оптимум Adaline на 2 шаге
Оптимум Hebb's rule на 9 шаге

![](https://github.com/limited1004/machine_learning/blob/master/r_studio/adaline_hebb_step.png)
