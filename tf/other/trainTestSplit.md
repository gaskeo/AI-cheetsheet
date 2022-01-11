# `train_test_split`

Функция для деления выборки на тренировочную и тестовую

```python
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(
    x=np.array,                   # массив с данными
    y=np.array,                   # массив с ответами
    test_size=float,              # процент в тестовой выборке [0, 1]
    train_size=float,             # в тренировочной
    random_state=int,             # random seed
    shuffle=bool                  # нужно ли перемешивать
)
```