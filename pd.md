# `pandas`

---
## `info` 
Информация о столбцах  
```
data.info()
```
---

## Индексы

```
data['column'] 
data[['column1', 'column2']]
data[data.columns[1, 4, 5]] 
```

---
## `corr`
Корреляция столбцов (насколько один столбец зависит от другого)
```
data.corr()
```
---
## `hist`
Построить гистограмму 
```
data['some column' ].hist()
```
