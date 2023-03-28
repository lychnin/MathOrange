## NPY格式
用于将数组保存到磁盘的一种简单格式，包含有关数组的全部信息。

.npy格式是numpy中的标准的二进制文件格式以持久化numpy数组的结构到磁盘。

.npz格式是持久化很多numpy数组到磁盘的标准格式。

一个.npz格式可以包括很多个.npy文件。

## pickle objects
pickle objects模块实现是对Python对象的序列化和反序列化的二进制协议。

pickling可以将Python层次结构转化为字节流。

unpickling是一个逆向操作。将字节流转化为Python层次结构。

pickle也可以被称为序列化、编组、扁平化等等。

## np.load
parameters:
file ：必须包含seek()和read()方法。序列化文件需要包括readline()方法。

mmap-mode：memory-map mode，使用给定的内存映射模式。内存映射对于访问大文件的小片段而不将整个文读入内存特别有用。

allow_pickle:允许加载数据存储为npy文件，可能因为安全性等原因禁用，但是禁用的话就会加载数据失败。

fix_imports:只有在将Python2的pickle数据加载时才有用，它会将Python2的旧名称转化为Python3的名称。

encoding:读取Python2字符串时使用的编码格式，只有在加载Python2生成的pickle文件时才有用，只能使用"ASCII"、"latin1"、"byte"。

max_header_size:header的最大大小

