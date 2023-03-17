学习链接
[深入浅出](https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E4%BA%8C%E7%AB%A0/2.1%20%E5%BC%A0%E9%87%8F.html#id2)
[官方教程](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
[w3c教程](https://www.w3cschool.cn/pytorch/pytorch-nr8s3bsu.html)
## Pytorch


Pytorch是torch的python版本，是由Facebook开源的神经网络框架，专门针对 GPU 加速的深度神经网络（DNN）编程。Torch 是一个经典的对多维矩阵数据进行操作的张量（tensor ）库，在机器学习和其他数学密集型应用有广泛应用。与Tensorflow的静态计算图不同，pytorch的计算图是动态的，可以根据计算需要实时改变计算图。但由于Torch语言采用 Lua，导致在国内一直很小众，并逐渐被支持 Python 的 Tensorflow 抢走用户。作为经典机器学习库 Torch 的端口，PyTorch 为 Python 语言使用者提供了舒适的写代码选择。

### 深度学习框架中的数据格式
#### 典型的有NCHW与NHWC
N-Batch(批)
C-Channel
H-Height
W-Width
![](../..\blog_picture\NCHW.png)

**逻辑表达上是4D数据**
在物理存储时，都是按照1D来存储的
![](../..\blog_picture\NCHW-physical.png)


NCHW是先取W方向数据，再取H方向，再C方向，最后N方向

NHWC是先取C方向，然后W方向，再H方向，最后N方向

以RGB图像为例
![](../..\blog_picture\RGB-nchw.png)

### tensor.mean()
可以计算不同维度的均值
## TensorFlow

一款优秀的、应用广泛的深度学习框架。


### Tensors
几何代数中的张量是基于向量与矩阵的推广
| 张量维度 | 代表含义     |
| -------- | ------------ |
| 0维张量  | 代表的是标量 |
| 1维张量  | 代表向量     |
| 2维张量  | 代表矩阵     |
| 3维张量  | 时间序列数据 |



类似于数组与矩阵的数据结构，使用这种数据结构来编码输入与输出。

这种数据结构可以运行在GPU上和特定的硬件上来加速计算。
```python
import torch
import numpy as np
```
#### 初始化Tensor
- 利用torch.rand()随机初始化矩阵
    import torch
    x=torch.rand(4,3)
    print(x)
- 全0矩阵的构建，可以通过torch.zeros()构造一个矩阵全为0并且通过dtype设置数据类型为long，除此之外,还可以通过torch.zeros()或者torch0zero_()和toech.zero_like()将现有的矩阵全部转换为全0矩阵
- 直接从数据中初始化
    ```python
    data=[[1,2],[3,4]]
    x_data=torch.tensor(data)
    ```
- 从numpy数组中初始化
    ```python
    np_array=np.array(data)
    x_np=torch.from_numpy(np_array)
    ```
- 根据其他Tensor初始化，若没有明确覆盖维度特征，会保持原Tensor特征
    ```python
    x_ones=torch.ones_like(x_data)
    print(f'Ones Tensor:\n {x_ones} \n')
    x_rand=torch.rand_like(x_data,dtype=torch.float)
    print(f'Random Tensor :\n {x_rand} \n')    
    ```
    返回值:
    ```python
    Ones Tensor:
    tensor([[1, 1],
        [1, 1]])

    Random Tensor:
    tensor([[0.3910, 0.9087],
        [0.8382, 0.3713]])
    ```
- 根据输入的维数
    ```python
    shape=(2,3,)
    rand_tensor=torch.rand(shape)
    ones_tensor=torch.ones(shape)
    zeros_tensor=torch.zeros(shape)
    ```
    
#### tensor 属性
```python
tensor=torch.rand(3,4)
print(f'Shape of tensor:{tensor.shape}')
print(f'Datatype of tensor:{tensor.dtype}')
print(f'Device tensor is stored on:{tensor.device}')
print(tensor)
```
返回值：
```python
Shape of tensor:torch.Size([3, 4])
Datatype of tensor:torch.float32
Device tensor is stored on:cpu
tensor([[0.9586, 0.6597, 0.3907, 0.7017],
        [0.5692, 0.8162, 0.7167, 0.3534],
        [0.0261, 0.4152, 0.4164, 0.8375]])
```

#### tensor 操作
有超过100种tensor操作
每一种都可以在GPU上执行(通常快于CPU)
##### 加法操作
```python
import torch
# 方式一
y=torch.rand(4,3)
x=torch.rand(4,3)
print(x+y)
# 方式二
print(torch.add(x,y))
# 方式三 in-place,原值修改
y.add_(x)
print(y)
```

##### 索引操作(类似于numpy)
- 维度变换
    张量的维度变换常见的方法有torch.view()与torch.reshape()
  - torch.view()
    利用这个函数返回的新tensor其实与源Tensor共享内存，更改其中的一个，另外一个也会跟着改变，==我们只是改变了对这个张量的观察角度==
  - torch.reshape()
    很多情况下，我们希望原始张量与变换后的张量互相不影响,为了使创建的张量与原始张量不共享内存，我们需要使用torch.reshape()
    ，也可以改变张量的形状，但是此函数不能保证返回的是拷贝值，所以官方不推荐使用，我们常先使用clone()创造一个张量副本，然后使用torch.view()进行维度变换

    ==note：使用clone还有一个好处,就是会被吉利在计算图中，即梯度回传到副本时也会传到源tensor==
    - 取值操作
    tensor.item()可以获得其value，而不获得其他性质
#### 广播机制
当对两个形状不同的tensor按元素运算时，可能会触发广播机制:先将两个tensor各自复制到相同维度的，然后再按元素相加

### timm 模型微调
timm是一个常见的预训练模型库，提供了众多计算机视觉的SOTA模型，可以当作torchvision的扩充版本。
#### 查看timm提供的预训练模型
```python
# 查看timm提供的预训练模型
import timm
avail_pretrained_models=timm.list_models(pretrained=True)
```
#### 查询同一系列不同方案的模型
每一种系列可能对应着不同方案的模型，比如Resnet模型包括了ResNet18、50、101等模型，我们可以在timm.list_models()传入想查询的模型的名称

```python
all_desnet_models=timm.list_models("*densenet*")
```
列表返回所有densenet系列的模型
```python
['densenet121',
 'densenet161',
 'densenet169',
 'densenet201',
 'densenet264',
 'densenet264d_iabn',
 'densenetblur121d',
 'tv_densenet121']
```

#### 查看模型参数
```python
model=timm.create_model('resnet34',num_class=10,pretrained=True)
model.default_cfg
```
返回结果
```python
{'url': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet34-43635321.pth',
 'num_classes': 1000,
 'input_size': (3, 224, 224),
 'pool_size': (7, 7),
 'crop_pct': 0.875,
 'interpolation': 'bilinear',
 'mean': (0.485, 0.456, 0.406),
 'std': (0.229, 0.224, 0.225),
 'first_conv': 'conv1',
 'classifier': 'fc',
 'architecture': 'resnet34'}
```

#### 使用和修改预训练模型
```python
model=timm.create_model('resnet34',pretrained=True)
```

查看某一层模型参数（以第一层卷积为例）
```python
model = timm.create_model('resnet34',pretrained=True)
list(dict(model.named_children())['conv1'].parameters())
```

#### 模型的保存
timm库所创建的模型是torch.model的子类，直接使用torch库中内置的模型参数保存、加载即可
```python
torch.save(model.state_dict(),'path')
model.load_state_dict(torch.load('path'))
```
### 自动求导
pytorch中，所有神经网络的核心是autograd包，该包为张量上的所有操作提供了自动求导机制，它是一个定义在运行时定义的框架，这意味着反向传播是根据代码如何运行来绝对的，并且每次迭代可以是不同的。

#### Autograd简介
tensor.Tensor是这个包的核心类
- 如果设置其属性.requires_grad为True,那么它就会==追踪对于该张量的所有操作==
- 当完成计算后可以通过调用.backward()来==自动计算所有的梯度==，如果 Tensor 是一个标量(即它包含一个元素的数据），则不需要为 backward() 指定任何参数，但是如果它有更多的元素，则需要指定一个gradient参数，该参数是形状匹配的张量
- 这个==张量的所有梯度将会自动累加到.grad属性==
- 要==阻止一个张量被跟踪历史==，可以调用.detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。为了防止跟踪记录(和使用内存),可以将内存包装在with torch.no_grad():中。在评估模型时特别有用，因为模型可能具有 requires_grad = True 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。
- Function类对autograd的实现也非常重要，它与tensor相互连接生成了一个无环图，编码了完整的计算历史，每个张量有一个属性grad_fn，该属性引用了创建tensor自身的function(除非这个张量是用户手动创建的，即这个张量的grad_fn是 None )

#### 梯度
tensor.grad会随着计算累加
tensor.grad.data.zero_()可以将grad清零
torch.autograd不能直接计算完整的雅可比矩阵，但是如果我们只想要雅克比向量积，只需将这个向量作为参数传给backward
```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
```

可以通过将代码块放在with torch.np_grad()中,来阻止autograd跟踪设置了.requires_grad=True的张量的历史记录
```python
with torch.no_grad():
    print((x ** 2).requires_grad)
```

如果我们想要修改tensor的数值，但是又不希望被autograd记录(即不影响反向传播)，那么可以对tensor..data操作

### 并行计算
利用PyTorch做深度学习过程中，肯能会遇到数据量较大无法在单块GPU上完成的场景

PyTorch可以编写模型让多个GPU参与训练，减少训练时间

#### CUDA
是NVID提供的GPU并行计算框架，对于GPU本身，使用的是CUDA语言来实现，但是，在编写深度学习代码时，使用的CUDA又是另一个意思，表示要求我们的模型使用GPU了。

在编写程序中，当我们使用了 .cuda() 时，其功能是让我们的模型或者数据从CPU迁移到GPU(0)当中，通过GPU开始计算。

==note==:
1.使用的是.cuda而不是.gpu是因为当前GPU的编程接口采用CUDA
2.数据在CPU与GPU之间传输比较耗时，应避免数据的切换
3.GPU虽然很快，但是在简单任务执行的时候要尽可能使用CPU
4.服务器有多个GPU时，我们应该指明使用的是哪一块GPU，否则默认使用第一块，可能会爆出out of memory的错误
```python
# 写在文件开始
import os
os.environ["CUDA_VISIBLE_DEVICE"]="2"#设置默认显卡
 CUDA_VISBLE_DEVICE=0,1 python train.py # 使用0，1两块GPU

```

#### 常见的并行方法
- 网络结构分布到不同的设备中，由于不同模型组件在不同CPU上GPU之间的传输很重要，在GPU这种密集任务中很难办到
- 同一层的任务分布到不同数据中
- 不同的数据分布到不同的设备中，执行相同的任务，这是现在的主流方式

#### 使用CUDA加速训练
在PyTorch框架下，CUDA的使用变得非常简单，只需要显式地将模型和数据通过.cuda()方法转移到GPU上就可加速我们的训练

## AI硬件加速设备
在进行模型部署与训练时，有时受限于CPU与GPU的性能，专用的AI芯片就显得尤为重要。

### CPU
CPU，即Central Processing Unit,中文名为中央处理器，它的功能是处理指令、执行操作、控制时间、处理数据

CPU对计算机的所有硬件资源进行控制调配，是计算机的运算与控制核心，计算机系统中所有软件层的操作最终都通过指令集映射为CPU的操作

CPU的主要职责其实并不是数据运算，还需要执行存储读取、指令分析、分支跳转等命令。深度学习算法通常需要进行海量的数据处理，用CPU执行算法时，CPU将花费大量的时间在数据/指令/的读取的分析上，而CPU的频率、内存等却不能无限制的提高，因此处理器的性能被限制。

### GPU
GPU为Graphics Processing Unit，中文名为图形处理单元，GPU的控制相对简单，大部分的晶体管可以组成各类专用电路，多条流水线，使得GPU的运算速度远高于CPU，同时GPU有了更加强大的浮点运算能力，可以缓解深度学习算法的训练难题。

但是GPU没有独立工作的能力，必须由CPU进行控制调用才能工作，且GPU的功耗一般较高。因此，随着AI的不断发展，高功耗低效率的GPU不再能满足AI训练的要求，因此速度更快，功能更单一的专用集成电路出现。

### 专用集成电路(Application-Specific Integrated Circuit,ASIC)是专用定制芯片，即为实现特定要求而定制的芯片，定制的特性有助于提高ASIC的性能功耗比，缺点是电路设计需要定制，相对开发周期长，功能难以扩展。

#### TPU
Tensor Processing Unit ,中文名为张量处理器，2015年谷歌在IO开发者大会上推出。

截止目前，谷歌已经发行了四代TPU芯片

芯片架构设计如下
![](../..\blog_picture\tpu.png)

整个TPU之中最重要的计算单元是矩阵乘单元:“Matrix Multiply Unit”，它可以在单个时钟周期中处理数十万次矩阵运算(Matrix),MMU有着与传统CPU、GPU截然不同的结构，称为脉动阵列，之所以称为脉动，是因为在这种结构中，数据一波一波地流过芯片，与心脏跳动供血的方式类似。

但在极大增加数据复用，降低内存贷款压力时，它也有两个缺点:
- 脉动矩阵主要实现向量/矩阵乘法，数据重排的额外操作增加了复杂性
- 在数据流经整个阵列后，才能输出结果，当计算的向量中元素过少，脉动阵列规模过大时，不仅难以将阵列中的每个单元利用起来，数据的导入与导出延时也随着尺寸扩大而增加。

##### 技术特点
- AI加速专用
- 脉动阵列设计
- 确定性功能与大规模片上尺寸
  - 传统GPU由于片上内存较少，因此在运行过程需要不断访问片外动态存取存储器，从而在一定程度上浪费了不必要的消耗。TPU比之GPU、CPU控制单元更小，给片上的运算单元提供了更大的面积。

#### NPU(Neural-network Processing Unit)
中文名为神经网络处理器，采用"数据驱动并行计算"的架构，特别擅长处理视频，图像类的海量多媒体数据。

从技术角度上看，深度学习实际上是一类多层大规模人工神经网络。

以寒武纪的DianNao架构为例简要介绍NPU

![](../..\blog_picture\DianNao.png)

基于神经网络的人工智能算法，是模拟人类大脑内部神经元的结构，上图中的neuron代表单个神经元,synapse代表神经元的突触。

## 思考
一项机器学习任务常常有以下的几个重要步骤：
- 首先是==数据的预处理==，其中重要的步骤包括数据格式的统一、异常数据的消除、必要的数据变换，同时==划分训练集、验证集、测试集==

- 之后是==模型选择==，并==设定损失函数和优化方法==以及==对应的超参数==(当然可以用sklearn中模型自带的损失函数和优化器)
- 最后==用模型去拟合训练集数据==，并==在验证集/测试集上计算模型表现==

$$数据的预处理划分\to训练集、验证集、测试集\to模型选择\to设定损失函数和优化方法\to对应的超参数\to用模型去拟合训练集数据\to在验证集/测试集上计算模型表现$$


深度学习与机器学习==在流程上类似==，但==在代码实现上有较大的差异==，首先，==由于深度学习所需的样本量很大==，一次加载全部数据运行可能会超出内存容量而无法实现，同时==还有批训练等提高模型表现的策略，需要每次训练读取固定的数量的样本送入模型中训练==，因此**深度学习在数据加载上需要有专门的设计**。

==在模型实现==上，深度学习与机器学习也有很大差异，由于深度神经网络层数较多，同时会有一些用于实现特定功能的层(如卷积层、池化层、批正则化层、LSTM层等)，==因此深度神经网络需要"逐层搭建"，或者预先定义好可以实现特定功能的模块，再把这些模块组装起来==，这种定制化的模型构建方式能够充分保证模型的灵活性，也对代码实现提出了新的要求。

然后是==损失函数与优化器的设定，要能够保证反向传播能够在用户自行定义的模型结构上实现==


上述步骤完成后，便可以开始训练，不过程序默认在CPU上运行，因此在代码实现中，需要把模型和数据放到GPU上运行，如果有多张GPU还要考虑模型和数据分配、整合的问题，最后还有一些指标需要放回CPU，涉及到一系列有关于GPU的配置与操作。

深度学习中训练和验证过程的最大的特点是读入数据是按批的，每次读入一个批次的数据，放入GPU中训练，然后将损失函数反向传播回网络最前面的层，同时使用优化器调整网络参数，这里会涉及到各个模块配合的问题，训练后还需要根据设定好的指标计算模型表现。

深度学习的差异总结:
- 深度学习在数据加载上需要有专门的设计
- 因此深度神经网络需要"逐层搭建"，或者预先定义好可以实现特定功能的模块，再把这些模块组装起来
- 损失函数与优化器的设定，要能够保证反向传播能够在用户自行定义的模型结构上实现
- 深度学习中训练和验证过程的最大的特点是读入数据是按批的

## 基本配置
对于一个PyTorch项目，需要导入一些Python常用的包来帮助我们快速实现功能，常见的包有os、numpy等,此外还需要调用PyTorch自身一些模块便于灵活使用，比如torch、torch.nn、torch.utils.data.Dataset、torch.utils.data.Dataloader、torch.optimizer

根据前面对深度学习任务的梳理，有如下几个超参数可以统一设置，方便后续调试时修改:
- batch size
- 初始学习率(初始)
- 训练次数(max_epochs)
- GPU配置

```python
# 批次的大小
batch_size=16
# 优化学习器的学习率
lr=le-4
max_epochs=100
```

除了直接将超参数设置在训练的代码里，我们也可以用yaml、json、dict等文件来存储超参数，这样便于后续的调试与修改，这种方式也是常见的深度学习库(mmdetection、Paddledetection、detectron2)和一些AI Lab里面比较常见的一种参数设置方式

我们的数据和模型如果没有经过显式指明设备，默认存储在CPU上，微加速模型训练，需要显式调用GPU
```python
# 方案一:使用os.environ,这种情况如果使用GPU不需要设置
import os
# 指明调用的GPU为0,1号
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

# 方案二:使用“device”，后续对要使用GPU的变量使用.to(device)即可
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 指明调用的CPU为1号
```

## 数据读入
PyTorch数据读入是通过Dataset+DataLoader的方式完成的，Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据。

我们可以定义自己的Dataset类来实现灵活的数据读取，这个类需要==继承PyTorch自身的Dataset类==，主要包含三个函数:
\__init__:用于向类中传入外部参数，同时定义样本集
\__getitem__:用于逐个读取样本集合中的元素，可以进行一定的变换,并将返回训练/验证所需要的数据
\__len__:用于返回数据集的样本数
```python
import torch
from torchvision import datasets
# ImageFolder类用于读取按一定结构存储的图片数据,data_transform可以对图像进行一定的变换、如翻转、裁剪等操作，可以自己定义
train_data = datasets.ImageFolder(train_path, transform=data_transform)
val_data = datasets.ImageFolder(val_path, transform=data_transform)
```

```python
class MyDataset(Dataset):
    def __init__(self, data_dir, info_csv, image_list, transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:,0]
        image_name = os.path.join(self.data_dir, image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_file)
```

上述Dataset就是一个自定义的Dataset

构建号后，就可以使用DataLoader来按批次读入数据了，实现代码如下:
```python

from torch.utils.data import DataLoader
train_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_siz,num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)

```
上述DataLoader的参数中:
- batch_size：样本是按“批”读入的，batch_size就是每次读入的样本数
- num_workers：有多少个进程用于读取数据，Windows下该参数设置为0，Linux下常见的为4或者8，根据自己的电脑配置来设置
- shuffle：是否将读入的数据打乱，一般在训练集中设置为True，验证集中设置为False
- drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练

```python
import matplotlib.pyplot as plt
images, labels = next(iter(val_loader))
print(images.shape)
plt.imshow(images[0].transpose(1,2,0))
plt.show()
```

## 模型构建
PyTorch中的神经网络构造一般是基于nn.moudle类的模型来完成的。

Moudle类是torch.nn模块里提供的一个模型构造类，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型。

