## pandas
```python
DataFrame.astype(dtype, copy=True, errors=’raise’, **kwargs)
```
作用：astype的作用便是转换数据类型
参数释义：
dtype:numpy.dtype或者Python类型
copy:copy=True时返回一个副本，copy=False时可能会传播给其他Pandas对象
errors:当数据无效时（无法转换为提供的dtype）是否允许报错
- raise:允许
- ignore:忽略
kwargs:传递给构造函数

dataframe.merge()

left_on,right_on同时使用时意味着取左侧基准列，右侧基准列值一样的行，
使用on时，针对的基准列是左右两个dataframe相同的列

left_index与right_index:按相同的索引组合

left_on与right_index:左侧基准列中的值与右侧索引值匹配则组合

left_index与right_on:左侧索引值与右侧基准列中的值匹配则组合

但这值得一提的是：列和索引混合使用后，得到的新列表的索引并不是以指定的索引列命名的，即混用只是在原dataframe中根据索引值或列进行值的匹配，新的索引从0开始计数，并且只有匹配成功的值才可以获得索引

```python
df.loc()
```
通过索引或者列名查找数据
还可以通过布尔值查找数据, 布尔值列表与index列表个数匹配，True表示显示对应的index行的值，False表示不显示