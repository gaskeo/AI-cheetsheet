# `numpy`

## Индексы
```python
# добавить одну размерность справа
data = ...
data = data[..., None] # (то же самое, что и data.reshape(-1, x, y, 1)) 
```

## `reshape`
Поменять размерность
```python
data = ...
data.reshape((50, 10))
```

## `flatten`
Все размерности выравнивает в одну: `(20, 40) -> (800, )`

## `np.argmax`
Индекс с максимальным значением
```python
import numpy as np
np.argmax(np.array([1, 3, 5, 2]), axis=0) # -> 2 
```