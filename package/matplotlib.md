matplotlib.pyplot.subplots(nrows=1, ncols=1, *, sharex=False, sharey=False, squeeze=True, width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, **fig_kw)

nrows:要画的子图行数
ncols:要画的子图列数
sharex：是否共享行坐标
sharey：是否共享列坐标
squeeze：如果为真，则将数组多余的维数舍弃
width_ratios：各个子图的行宽比例
height_ratios：各个子图的高度比例
subplot_kw：子图样式
gridspec_kw：子图网格样式
fig_kw：额外的参数

返回：
fig：画布
ax：返回一个轴对象或者一个轴对象的数组

%matplotlib inline作用:
表示把图表嵌入笔记中，使用 %matplotlib inline

backend和frontend
matplotlib在完成各种绘图输出时，用户面向的代码是frontend，而在幕后完成那些复杂工作的是backend。

