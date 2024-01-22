
# Vectorization

Геометрические операции в mmcore могут быть представлены в неизменном, а также векторизированом виде.

<tldr>


Векторизация позволят вам применить к одним и тем же данным операции для работы с
точками, кривыми, граничными и прочими представлениями.

Например, сетка может быть представлена как набор полигонов,
полигонов как набор точек, точка как набор координат и т.д. Вы можете работать с массивом float любого шейпа
как с массивом точек, чисел, полилиний в зависимости от операции которую вы применяете.

</tldr>

## Отличия векторизированых функций

Отличия векторизированых функций лучше всего рассматривать на примере.

```python
import numpy as np
from mmcore.geom.aabb import aabb, aabb_vectorized
```

### 1D example

Создадим массив из 10 3D случайных точек

```python
>>> points=np.random.random((10,3))
>>> points.shape
 (10, 3)
```

Вычислим AABB для этого массива.

```python
>>> res = aabb(points)
>>> res.shape 
(2, 3)
>>> res
array([[0.03442901, 0.05956184, 0.00195626],
       [0.73280618, 0.96770464, 0.90464465]])
       
>>> res = aabb_vectorized(points)
>>> res.shape 
(2, 3)
>>> res
array([[0.03442901, 0.05956184, 0.00195626],
       [0.73280618, 0.96770464, 0.90464465]])
       
```

Размеры, а также порядок элементов идентичны для обоих массивов. Это ожидаемо,
так-как для AABB скалярным аргументом
является плоский массив точек.

### ND example

Рассмотрим пример с большей размерности.

Передадим на входы массивы c `ndim` равным 3 и 5.




 <compare first-title="Not vectorized" second-title="Vectorized" >
 <code-block lang="python" prompt=">>>,>>>,>>>,>>>,Out:,>>>,>>>,>>>,Out:">
 shp=(7, 10, 3)
 points=np.random.random(shp)
 res=aabb(points)
 res.shape
 (2, 7, 3)
 shp=(5, 6, 7, 10, 3)
 points=np.random.random(shp)
 aabb(points).shape
 (2, 5, 6, 7, 3)
 #Первое измерение возвращаемого объекта стало первым для массива

 </code-block >
 <code-block lang="python" prompt=">>>,>>>,>>>,>>>,Out:,>>>,>>>,>>>,Out:">
 shp=(7, 10, 3)
 points=np.random.random(shp)
 aabb_vectorized(points)
 res.shape
 (7, 2, 3)
 shp=(5, 6, 7, 10, 3)
 points=np.random.random(shp)
 aabb_vectorized(points).shape
 (5, 6, 7, 2, 3)
 # Порядок как и на входе 
 </code-block >
 </compare>

### Summary

#### Signatures

- `groups` - items in n-kind groups `group 0, group 1, ... group n, group n+1`
- `points` - count of points in last group
- `xyz` - point coordinates
- `min-max` - min and max AABB value per dimention

Сигнатуры "скалярных" объектов а также наборов данных на вход выход:

|            |        **input**         |   **output**   | 
 |------------|:------------------------:|:--------------:|
| **object** |      `points, xyz`       | `min-max, xyz` | 
| **data**   | `groups..., points, xyz` |       ?        |

Общая сигнатурв входа и выходы для обоих случаев:

|             | **first** | **second** | **last** |
 |-------------|-----------|------------|----------|
| **input**   | groups    | points     | xyz      |
| **not vec** | min-max   | groups     | xyz      |
| **vec**     | groups    | min-max    | xyz      |

---

## Implementation

Векторизация выполняется с
использованием <code>numpy.vectorize</code> или ее оберткой <code>vectorize</code> из модуля <code>mmcore.func</code>
которая предоставляет ряд дополнительных удобств.


