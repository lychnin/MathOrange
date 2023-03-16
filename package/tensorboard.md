TensorBoard 是包含在TensorFlow中的一个子服务。

安装:
它包含在了TensorFlow中，也可以单独安装:
```python
pip install tensorboard
```

启动TensorBoard

```python
tensorboard --logdir=<directoy_name>
```

在Jupyter Notebooks中使用TensorBoard

```python
%load_ext tensorboard
```

```python
%tensorboard --logdir logs
```

使用TensorBoard
- 本地使用
    在模型编译后，我们需要创建一个回调并在调用 fit 方法时使用。
    ```python
    tf_callback=tf.keras.callbacks.TensorBoard(log_dir="./logs")
    ```
    ```python
    model.fit(X_train,y_train,epochs=5,callbacks=[tf_callback])

    ```

仪表盘
- Scalars
    
