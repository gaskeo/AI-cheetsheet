# `Tokenizer`

Чем чаще число — тем меньше токен

### Создание токенизатора и обучение на тексте 
```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(
    num_words=int,          #   максимальное количество слов для токенизации
    filters=str,            #   символы, которые не нужны
                            #   '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
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