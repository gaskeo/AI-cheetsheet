# Модель
`tensorflow.keras.models`

```python
import numpy as np
model = ...

model.compile(optimizer=..., loss=..., metrics=...)         # скомпилировать сеть
model.fit(                                                  # обучить сеть
    x_train=np.array, 
    y_train=np.array, 
    epochs=int, 
    validation_split=[0-1] # сколько на проверочную выборку
)             
model.evaluate(x_test=np.array, y_test=np.array)            # проверить на тестовой выборке
model.predict(one_elem=np.array)                            # предсказать по 1 элементу
                                                            # размерность (1, (размерность самого элемента))
model.summary()                                             # информация о модели
```

## `Sequential`
Простейшая полносвязная сеть. 
```python
from tensorflow.keras.models import Sequential

from typing import Iterable

Sequential(
    layers=Iterable    #   слои нейронной сети
)
```

## `Model` (functional api)
Другой способ задания модели
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
#            concatenate((10 + 102))
#                    |
#                 Dense(15)
```

# Слои
`tensorflow.keras.layers`

## `Dense` 
Самый простой полносвязный слой (каждый предыдущий нейрон связан с каждым следующим)
```python
from tensorflow.keras.layers import Dense

Dense(
    units=int,          #   размерность выходного слоя
    activation=str,     #   функция активации
    use_bias=bool       #   использование bias (t)
)
```

## `BatchNormalization`
Преобразует входные данные в числа от `-1` до `1`

## `Flatten`
Выравнивает входные данные до одной размерности: `(64, 10) -> (640, )`

## `Dropout`
Выключает случайные нейроны
```python
from tensorflow.keras.layers import Dropout

Dropout(n=float)  #  n - какой процент нейронов надо отключить (0, 1)
```

## `Conv2D`
Конволюционный слой
```python
from tensorflow.keras.layers import Conv2D
import numpy as np

Conv2D(
    filters= int,           #   количество фильтров в слое 
                            #   (количество нейронов в нейронном слое)
                            #
    input_shape=np.array,   #   только на входном слое, 
                            #   каждый пиксель оформлен отдельно:
                            #   (28, 28, 1) - картинка 28x28 + каждый пиксель - в своем массиве
                            #
    kernel_size=(int, int), #   размер конволюционного фильтра (t)
                            #   (3, 3), (5, 5), ...
                            #
    activation=str,         #   функция активации
    padding=str,            #   либо same, либо valid
                            #   same - на выходе такое же по размеру изображение
                            #   valid - края фильтра не заходят за края изображения, 
                            #           крайние пиксели обрезаются (размер фильтра // 2)
                            #
    strides=int or tuple,   #   шаг прохода фильтра: прыжок между пикселями
                            #   если tuple: (x, y) - шаг по x, шаг по y 
)
```

## `MaxPool2D`
Слой для уменьшения размеров изображения за счет выбора из соседних пикселей пикселя с максимальным значением 
```python
from tensorflow.keras.layers import MaxPool2D

MaxPool2D(
    (2, 2)  # размер окна, уменьшающего изображение
)
# +---+---+
# | 1 | 2 |                           +---+
# +---+---+ ---> Max(1, 2, 5, 6) ---> | 6 |
# | 5 | 6 |                           +---+
# +---+---+

```

## `Embeding`
Слой для генерации текста 
```python
from tensorflow.keras.layers import Embedding

Embedding(
    input_dim=int,      # максимальное количество слов
    output_dim=int,     # выходной слой
    input_lenght=int,   # количество слов в одном мешке
)
```

## `SpecialDropout`
Отключает столбик весов, по параметрам как `Dropout`

## `SimpleRNN`
Слой с рекуррентной нейронной сетью `t` 
```python
from tensorflow.keras.layers import SimpleRNN

SimpleRNN(
    units=int,                  #  количество внутренних нейронных сетей (количество выходов)
    dropout=float,              #  для входов
    recurrent_dropout=float,    #  для повторяющегося состояния
    activation=str,   
)
```

## `LSTM` `t` 
```python
from tensorflow.keras.layers import LSTM

LSTM(
    units=int,              #  как 'SimpleRNN'
    return_sequences=bool   #  возвращать промежуточные значения -> 
    # -> можно делать каскады LSTM слоев Sequential([
    #                                                   LSTM(), 
    #                                                   LSTM()
    #                                                ])
)  
```

## `Conv1D`
```python
from tensorflow.keras.layers import Conv1D

Conv1D(
    filters=int,        #  количество фильтров в слое
    kernel_size=int,    #  размер окна
    activation=str, 
)
```

## `Reshape`
Слой, подгоняющий размерность данных

```python
from tensorflow.keras.layers import Reshape

Reshape((20, 10))  
```

# Функции активации

## `Sigmoid`
Возвращает значения от `0` до `1`
* обычно применяется на выходном слое
* задача бинарной классификации (принадлежит/не принадлежит)

## `Relu`
Значения меньше нуля приравниваются к нулю
* обычно в скрытых слоях
* используется с картинками (`RGB`)

## `Tanh` - гиперболический тангенс
Более плавный по сравнению с `Sigmoid`

Возвращает значения от `-1` до `1`
* лучше работает со значениями, близкими к нулю
* можно использовать с картинками (`LAB` формат)

## `softmax`
Распределяет вероятность между классами
* обычно на последнем слое 
* в задачах классификации

# Функции ошибок

## `mae` - mean absolute error
Средняя абсолютная ошибка - среднее отклонение точки от предсказанного значения
* для регрессии

## `mse` - mean square error
Среднеквадратичная ошибка - отклонение в квадрате
* для регрессии

## `binary_crossentropy`
* для бинарной классификации для `[0, 1]`

## `categorical_crossentropy`
* для классификации, когда в выходном слое массив с индексом правильного элемента  

## `categorical_accuracy`
* для классификации

## `sparse_categorical_crossentropy`
* для классификации, когда в выходном слое только один нейрон с `id` класса

# Оптимизаторы `t`

## `adam`
Один из лучших оптимизаторов
```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(
    learning_rate=float   #   шаг обучения (по стандарту 10 ** -5)
)
```


# callbacks
Коллбэки нужны для отслеживания состояния модели на эпохе

```python
import tensorflow as tf

class MyCallBack(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs): 
        if logs['loss'] < 90: 
            print('конец') 
            self.model.stop_training = True


model = ...
...
model.fit(..., callbacks=[MyCallBack()])
```


# Прочее


## `train_test_split`
Функция для деления выборки на тренировочную и тестовую
```python
from sklearn.model_selection import train_test_split
import numpy as np

x_train, x_test, y_train, y_test = train_test_split(
    x=np.array,               # массив с данными
    y=np.array,               # массив с ответами
    test_size=float,       # процент в тестовой выборке [0, 1]
    train_size=float,      # в тренировочной
    random_state=int,      # random seed
    shuffle=bool           # нужно ли перемешивать
)
```


## `plot_model`
Функция для представления схемы нейронной сети
```python
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)
```

## `to_categorical` 
Функция для представления значений по категориям 
```python
from tensorflow.keras.utils import to_categorical
import numpy as np

a = np.array([1, 5, 2, 5, 2])
to_categorical(a) # -> 
# -> array(
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 0, 1]
)
```

## `Tokenizer` (следующие 3 блока кода)
Чем чаще число — тем меньше токен
```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(
    num_words=int,          #   максимальное количество слов для токенизации
    filters=str,            #   символы, которые не нужны
    lower=bool,             #   преобразовать в нижний регистр
    split=str,              #   разделитель слов
    oov_token=str,          #   как помечать слова, выходящие за num_words ('unknown')
    char_level=bool         #   если да, то токенизация на уровне символов 
)

train_text = ['text1', 'text2', 'text3', ...]
tokenizer.fit_on_text(train_text)                             # обучение токенизатора на тренировочной выборке
train_text_indexes = tokenizer.texts_to_sequences(train_text) # преобразование текстов в последовательность токенов
```
### Подготовка текстовых данных для классификации
```python
from tensorflow.keras.utils import to_categorical

import numpy as np
from typing import List

# нарезка выборки
def cut_words(word_indexes: List[int], cut_len, step): 
    """
    cut_len = 3 
    step = 2
    
              samples[1]
         index(2)  index + cut_len(5)
     step  \             /
    +-----++------------+
    [1, 54, 234, 245, 24, 64, 2, 257, 27, 7, 4, 412, 11] -> [[1, 54, 234], [234, 245, 24], ...]
    +----------+         
   /            \
index(0)   index + cut_len(3)
     samples[0]
     
    """
    
    samples = []
    word_count = len(word_indexes)
    index = 0
    while index + cut_len <= word_count: 
        samples.append(word_indexes[index:index + cut_len])
        index += step
    return samples
    
 
def create_samples(text_indexes: List[List[int], ...], cut_len: int, step: int) -> (np.array(), np.array()): 
    """
    создает пару (x, y) из текстов, 
    где x - набор текстов вида [[51, 15, 542, 6], [24, 252, 245, 52]]
                                +--------------+  +----------------+
                                     cut_len            cut_len
                                     
    и y - бинарно расщепленные категории: [[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]]
                                           +----------------+  +----------------+
                                             classes_count       classes_count
    
    возвращает 2 массива, элемент i первого - токенизированный текст, 
    а элемент i второго - категория, к которой относится текст 
    """
    
    classes_count = len(text_indexes)
    word_samples_by_classes = [cut_words(words, cut_len, step) for words in text_indexes]
    x_samples, y_samples = [], []
    for category in range(classes_count): 
        for words in word_samples_by_classes: 
            x_samples.append(words)
            y_samples.append(to_categorical(category, classes_count))
    return np.array(x_samples), np.array(y_samples) 
```
### Объединяем (делаем мешок слов)
```python
x_train, y_train = create_samples(train_text_indexes, 1000, 100)
x_train_binary = tokenizer.sequences_to_matrix(x_train.tolist())
```

## `StandardScaler`
нормализатор

```python
from sklearn.preprocessing import StandardScaler

x_train = ...
scaler = StandardScaler()
scaler.fit(x_train)
x_train_skaled = scaler.transform(x_train)

```