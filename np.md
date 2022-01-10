# `numpy`

## Индексы

```python
# добавить одну размерность справа
import numpy as np

data = np.array([1, 2, 3])

data[0:2]  # -> [1, 2]
data[:2]   # -> [1, 2]
data[:-2]  # -> [3]

data[data <= 2]  # -> [1, 2]

data[(data <= 2) & (data % 2 == 0)] # -> [2] (для или используется |)

data = data[:, None]  # (то же самое, что и data.reshape(-1, x, y, 1))
# np.newaxis == None
data[:, np.newaxis]  # -> [[1], [2], [3]] 
                     # -> взять исходную размерность 
                     # и добавить размерность в последнюю размерность (во внутреннюю)
data[None, :]   # -> [[1, 2, 3]] -> добавить внешнюю размерность

data2 = np.array([[1, 2, 3], [4, 5, 6]])
data2[                # то есть из первой размерности взять элементы  
    ([0, 1], [0, 2])  # под индексами 0 и 1, а из второй размерности элементы
]                     # под индексами 0 и 2
```

## Базовые операции
```python
import numpy as np

a = np.array([1, 2])
b = np.array([3, 1]) 
a + b  # -> np.array([1 + 3, 2 + 1])  -> np.array([4, 3])
# операции происходят поэлементно 

a * 1.5 # -> [1 * 1.5, 2 * 1.5] -> [1.5, 3.]
```

## Сортировка 
```python
import numpy as np

data = np.array([2, 45, 5, 6, 244])
np.sort(data)   # -> отсортированный массив 
np.argsort(data, axis=0)  # -> отсортированный массив в axis'ной размерности
```

## `concatenate`
Объединить массивы с одинаковым количеством размерностей 
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.concatenate((a, b), axis=0)  # axis - на какой глубине объединять
```

## `ndim` `size` `shape`
`ndim` - количество размерностей 
`size` - количество элементов
`shape` - геометрия массива

## `reshape`
Поменять размерность
```python
import numpy as np

data = np.array([1, 2, 3, 4])
data.reshape(2, 2)  # -> [[1, 2], [1, 2]]
data.reshape(data.shape, 1)     # -> [[1], [2], [3], [4]] 
```

## `sum`
Сумма элементов
```python
import numpy as np

a = np.array([1, 2, 3])  
a.sum()     # -> 6

b = np.array([[1, 2], [3, 4]])
b.sum()         # -> 10 [1 + 2 + 3 + 4]
b.sum(axis=0)   # -> np.array([4, 6])   [1 + 3, 2 + 4]  
b.sum(axis=1)   # -> np.array([3, 7])   [1 + 2, 3 + 4]
```

## `max` `min`
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
a.max()  # -> 4 
a.max(axis=1)   # -> [2, 4]
a.min(axis=0)   # -> [1, 2] 
```

## `flatten`
Все размерности выравнивает в одну: `(20, 40) -> (800, )`

## `np.argmax`
Индекс с максимальным значением
```python
import numpy as np
np.argmax(np.array([1, 3, 5, 2]), axis=0) # -> 2 
```

## `np.nonzero`
Индексы подходящих элементов
```python
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6]])
nz = np.nonzero(data < 5)  # -> (
#                                 array([0, 0, 0, 1]),
#                                 array([0, 1, 2, 0])
#                               )
# индексы во внешнем слое и индексы во внутреннем 
# -> data[nz[i]][nz[i]] == i-ый элемент массива data, подходящий под условие
data(nz) # -> array(1, 2, 3, 4) -> все элементы, подходящие под условие
```