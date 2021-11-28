# `pandas`

## Индексы
```python
data = ...
data = data['column'] 
data = data[['column1', 'column2']]
data = data[data.columns[1, 4, 5]] 
```

## `info` 
Информация о столбцах  
```python
data = ...
data.info()
```

## `corr`
Корреляция столбцов (насколько один столбец зависит от другого)
```python
data = ...
data.corr()
```

## `hist`
Построить гистограмму 
```python
data = ...
data['some column' ].hist()
```
