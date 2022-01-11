# `StandardScaler`
Нормализатор данных 

```python
from sklearn.preprocessing import StandardScaler

x_train = ...
scaler = StandardScaler()
scaler.fit(x_train)
x_train_sсale = scaler.transform(x_train)

```