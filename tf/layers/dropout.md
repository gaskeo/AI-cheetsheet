# Dropout

Слой, выключающий случайные нейроны, служит для предотвращения переобучения

```python
from tensorflow.keras.layers import Dropout

Dropout(
    rate=...  # какой процент нейронов нужно отключить от 0 до 1
)
```