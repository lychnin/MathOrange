我们可以在一个新的python模块上工作，并在jupyter环境中测试它，但是当模块发生变化时，必须在笔记本环境中重新加载模块。

一种简单的解决方案:使用autoreload来确保使用的是模块的最新版本。

```Python
%load_ext autoreload
%autoreload 2
```
|               |                                    |
| ------------- | ---------------------------------- |
| %autoreload 0 | 不执行重新加载命令。               |
| %autoreload 1 | 只重新加载所有                     | %aimport 要加载的模块 |
| %autoreload 2 | 重新加载 除了%aimport 要加载的模块 |
| %aimport      | 列出要自动加载或不自动加载的模块。 |
| %aimport foo  | 自动加载模块 foo                   |
| %aimport -foo | 不自动加载模块foo                  |
