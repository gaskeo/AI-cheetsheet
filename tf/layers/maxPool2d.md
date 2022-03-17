# MaxPool2D

---
### `t` 
* [конволюционная нейронная сеть](../../terms.md#convolutional-neural-network)
---

Слой для уменьшения размеров изображения за счет выбора из соседних пикселей максимального значения

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