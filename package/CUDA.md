graphics processing unit

GPU是显卡上的一块芯片

![image-20230111203158048](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230111203158048.png)

显示接口卡，显示适配器

数模信号转换，输出显示图形

独立显卡比同期的集成显卡好，独立显卡有自己的显存，集成显卡没有自己的显存，使用物理内存

在没有GPU之前，所有任务都交给CPU来做，但有了GPU之后，二者进行分工，逻辑性强的事物处理与串行计算交给CPU来做，GPU专注于执行高度线程化的并行处理任务（大规模计算任务），GPU不是独立的平台，必须与CPU协同工作，可以看成是CPU的协处理器，因此当我们说GPU并行计算时，实际说的是CPU+GPU的异构计算架构。他们通过PCLE总线连接在一起，CPU所在位置成为主机端，GPU所在位置称为设备端。

GPU包含更多的计算核心，特别适合数据并行的计算密集型任务，如大型矩阵运算，而CPU的运算核心较少，但是可以实现复杂的逻辑运算，更适合控制密集型的任务，此外，CPU的线程是重量级的，上下文切换开销大，但GPU由于存在很多核心，线程是轻量级的，因此，两者可以互补，CPU负责逻辑复杂的串行程序，GPU重点处理数据密集型的并行计算程序，==GPU只是替CPU分担工作，而并不是为了取代==

CUDA(Compute Unified Device Architecture)

2006年NVIDA发布：建立在GPU上的通用并行计算平台和编程模型，提供了GPU编程的简易接口，基于CUDA编程可以构建基于 GPU计算的应用程序，利用GPU的并行计算引擎来更加高效地解决比较复杂的计算难题。

cudnn则是针对深度卷积神经网络的加速库

CUDA在软件方面组成有：CUDA库，应用程序编程接口（API），运行库，两个通用数学库（CUFFT、CUBLAS），另一方面，CUDA提供了片上共享内存，使线程之间可以共享数据。

## 编程模型

CUDA架构中引入主机（host）和设备（device）的概念。

CUDA程序中既包含host程序，又包含device程序，同时，host与device间可以进行通信。

host：CPU与系统的内存

device：GPU以及本身的显存

DRAM(动态随机存取储存器)：最为常见的系统内存，DRAM只能将数据保存很短的时间，使用电容存储，每隔一段时间刷新一次，如果存储单元没有被刷新，存储的信息就会丢失。

## CUDA程序执行流程

1.分配host内存，进行数据初始化

2.分配device内存，并从host将数据拷贝到device上

3.调用CUDA的核函数在device上完成计算

4.将device上的运算结果拷贝到host上

5.释放device与host上分配的内存

## 线程层次结构

## kernel

CUDA执行流程中最重要的一个过程是调用CUDA的核函数来执行并行计算，kernel是CUDA中的一个重要的概念：

在CUDA的程序架构中，主机端代码部分在CPU上执行，是普通的C代码，当遇到数据并行处理的部分，CUDA便会将程序编译为GPU可以执行的程序，传送给GPU，这个程序就是kernel，设备端代码部分在GPU上执行，此代码部分在kernel上编写（.cu文件），kernel用__ global __符号声明，在调用时需要用<<<grid,block>>>来制定kernel要执行的结构

CUDA是通过函数类型限定词区别在host和device上的函数，主要的三个函数类型限定词如下：

global：在device上执行，从host中调用

device：在device上执行，只可以从device中调用，不可以和global同时用

host：在host上执行，仅可以从host上调用，一般省略不写，不可以和global同时用，可以和device同时用，此时函数会在device和host上都编译

## grid

kernel在device上执行时，实际是启动了很多个线程，一个kernel所启动的所有线程称为一个grid，同一个grid的线程共享相同的全局内存空间，grid是线程结构的第一层次。

## block

grid可以分为很多个线程块（block），一个block owns many 线程（thread）,every blocks was runned parelledly ,blocks can not communicate each other ,they dont't own order which one was runned first.There is a limit  that the numbers of blocks can not surpass 65535

属于结构的第二层次

grid和block都是定义为dim3类型的变量，dim3可以看成是包含三个无符号整数（x，y，z）成员的结构体变量

CUDA中，每一个线程都要执行核函数，每一个线程需要kernel的两个内置坐标变量（blockIdx，threadIdx）来唯一标识，其中blockIdx指明线程所在grid中的位置，threaIdx指明线程所在block中的位置。它们都是dim3类型变量。

每个block有包含共享内存（Shared Memory）,可以被线程块中所有线程共享，其生命周期与线程块一致。

## thread

一个CUDA的并行程序会被多个threads来执行，多个threads会被组成一个block，同一个block中的thread可以同步，也可以通过shared memory通信。

## 线程束 warp
GPU执行程序时的调度单位，SM的基本执行单元。目前在CUDA架构中，warp是一个包含32个线程的集合，这个线程集合被“编织在一起”并且“步调一致”的形式执行。同一个warp中的每个线程都将以不同数据资源执行相同的指令，这就是所谓 SIMT架构(Single-Instruction, Multiple-Thread，单指令多线程)。

## CUDA的内存模型

**SP**:最基本的处理单元，streaming processor，也称为CUDA core。最后具体的指令和任务都是在SP上处理的。GPU进行并行计算，也就是很多个SP同时做处理。

**SM**：GPU硬件的一个核心组件是流式多处理器（Streaming Multiprocessor）。SM的核心组件包括CUDA核心、共享内存、寄存器等。

一个kernel实际会启动很多线程，这些线程是逻辑上并行的，但是网格和线程块只是逻辑划分，SM才是执行的物理层，在物理层并不一定同时并发

对cuda有要求，cuda与nvidia驱动有相关关系，cudnn依赖于cuda