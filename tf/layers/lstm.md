# LSTM `t`

---
### `t` 

* [LSTM](../../terms.md#lstm)

---

```python
from tensorflow.keras.layers import LSTM

LSTM(
    units=int,                  #  количество внутренних нейронных сетей (количество выходов)
    dropout=float,              #  для входов
    recurrent_dropout=float,    #  для повторяющегося состояния
    activation=str,   
)
```