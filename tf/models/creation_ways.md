# Способы создания модели

## `Sequential`
Для создания обычной полносвязной сети

```python
from tensorflow.keras.models import Sequential

Sequential(
    layers=...  # массив со слоями сети
)
```

## `Model`
Функциональное API для создания сложных сетей
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, Flatten

input1 = Input((20, 10))
input2 = Input((100, 15))

x1 = Flatten()(input1)
x1 = Dense(100)(x1)
x1 = Dense(10)(x1)

x2 = Flatten()(input2)
x2 = Dense(102)(x2)

x = concatenate([x1, x2])
x = Dense(15)(x)

model = Model((input1, input2), x)
#   Input((20, 10))      Input((100, 15))
#         |                     |
#      Flatten                  |               
#         |                  Flatten  
#     Dense(100)                |   
#         |                     |
#     Dense(10)              Dense(102)
#          \                   /
#           +-----------------+   
#                    |
#           concatenate((10 + 102))
#                    |
#                 Dense(15)
```
