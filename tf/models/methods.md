# Модели
`tensorflow.keras.models`

Модель — сама нейронная сеть, состоящая из слоев 

## Основные методы модели
### `compile`
Компиляция модели

```python
model = ...
model.compile(optimizer=...,   # оптимизатор, подробнее в ...  
              loss=...,        # функция потерь, подробнее в ..., 
              metrics=...      # метрики, подробнее в ..., 
)
```

### `fit`
Обучение модели

```python
import numpy as np

model = ...
model.fit(
    x_train=np.array, 
    y_train=np.array, 
    epochs=int, 
    validation_split=[0-1] # в каком соотношении x_train 
    # разделится на тренировочную и выборку для валидации  
)
```

### `evaluate`
Проверка модели на тестовой выборке

```python
import numpy as np

model = ...
model.evaluate(x_test=np.array, y_test=np.array)
```

### `predict` 
Предсказание по входным данным

`model.predict(x)`

### `summary`
Информация о модели

`model.summary()`
