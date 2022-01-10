# SimpleRNN

---
### `t` 
* [рекуррентная нейронная сеть](../../terms.md#rnn)

---

Слой с рекуррентной `t` нейронной сетью, создает сеть внутри сети

```python
from tensorflow.keras.layers import SimpleRNN

SimpleRNN(
    units=int,                  #  количество внутренних нейронных сетей (количество выходов)
    dropout=float,              #  для входов
    recurrent_dropout=float,    #  для повторяющегося состояния
    activation=str,   
)
```