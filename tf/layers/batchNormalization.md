# BatchNormalization

---
### `t`
* [batch](../../terms.md#batch)
---

Преобразует входные данные в пределах одного батча в отрезок от `-1` до `1`

```python
from tensorflow.keras.layers import BatchNormalization

BatchNormalization(
    axis=... # на какую размерность применять нормализацию 
)
```