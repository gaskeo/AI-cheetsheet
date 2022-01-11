# `to_categorical`

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
# )
```