# Conv2D

### `t` 
* [конволюционная нейронная сеть](../../terms.md#convolutional-neural-network)
* [конволюционный фильтр](../../terms.md#convolutional-filter)
---

Аналогично [`Conv2D`](conv2d.md), только для текста

```python
from tensorflow.keras.layers import Conv1D

Conv1D(
    filters=int,        #  количество фильтров в слое
    kernel_size=int,    #  размер окна
    activation=str, 
)
```