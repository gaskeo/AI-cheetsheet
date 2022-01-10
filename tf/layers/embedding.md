# Conv2D

Слой, отлично подходящий для генерации текста

```python
from tensorflow.keras.layers import Embedding

Embedding(
    input_dim=int,      # максимальное количество слов
    output_dim=int,     # выходной слой
    input_lenght=int,   # количество слов в одном мешке
)
```