# AI

## GAN - 生成式对抗网络 generative adversarial network  
> 有一个【画手】想要伪造名画大师的真迹去骗钱。但即使不断地练习绘画，他的画还是能看出是假的。于是，他找到一个【艺术鉴赏专家】检查他的作品，告诉他从哪里看出这幅画是赝品。于是，通过不断的纠错和修改，直到大师也无法判断画手的作品是否是赝品后，画手成为了有钱人
- GAN 有 2 个部分，即生成器网络和判别器网络，它们协同工作以产生与真实内容雷同的输出   
- 【画手】 - 生成器网络（generator network）：负责从任意输入开始生成逼真的图像
- 【艺术专家】- 判别器网络(discriminator network):判断给定的图像是由生成器生成的还是一幅真实的图像
- 绘画创作和检测赝品互相 *“对抗”* 。
> 对抗（adversarial）意味着这 2 个网络在竞争，其中一个要比另一个更聪明，而网络意义就显而易见了  
- 生成器的最终目标是欺骗判别器，混淆真伪图像  
- 判别器的最终目标是发现它何时被欺骗了，同时告知生成器在生成图像中可识别的错误。  
- 2 个网络都是基于彼此网络的结果进行训练的，并推动彼此对网络参数进行优化  

> 生成器负责从任意输入开始生成与目标近似的结果，由判别器判断生成的数据与目标是否一致。随着训练的推进，信息从判别器返回，而生成器使用这些信息进行改进。在训练结束时，生成器可以生成以假乱真的图像了，而判别器却不再能够识别出图像的真伪了

![image](https://user-images.githubusercontent.com/64322636/219339856-990f8b97-48c3-410d-92ff-1a57d011a7f6.png)
- 判别器与生成器之间谁获胜其实没有意义

- 2014年Ian Goodfellow 发表的<Generative Adversarial Network>(https://arxiv.org/abs/1406.2661) 将生成式对抗网络引入到深度学习领域中

- 生成网络产生“假”数据，并试图欺骗判别网络 
- 判别网络对生成数据进行真伪鉴别，试图正确识别所有“假”数据
- 在训练迭代的过程中，两个网络持续地进化和对抗，直到达到平衡状态(纳什均衡），判别网络无法再识别“假”数据，训练结束。

## CycleGan循环生成式对抗网络【稳定】
- CycleGAN使用一种支持在2种不同类型的图像之间来回转换的架构   
> 可以将一个领域的图像转换为另一个领域的图像，而不需要我们在训练集中显式地提供匹配对 
- 一个经过训练可以欺骗 2 个判别器网络的CycleG  
- 比如把斑马转换为马【斑马🦓 -> 棕马🐎】  
![image](https://user-images.githubusercontent.com/64322636/219341287-3bcf77de-37ea-4871-a12b-89fed2d196a3.png)
- 第 1 个生成器: 学习从属于不同分布域的图像（本例是马），生成符合目标域的图像（本例是斑马）。
- 因此判别器无法分辨出从马的照片中产生的图像是否真的是斑马的图像
- 产生的假斑马图像通过另一个生成器发送到另一个判别器，由另一个判别器来判别

#### 深度伪造
- 面部交换技术

### VAE生成模型
## torch
- 深度学习开发框架之一
> pytorch是一个python编程库，有助于构建深度学习项目（深度学习库)  
> PyTorch 库允许你高效地构建和训练神经网络模型  
> pytorch为了性能是用c++和cuda编写的  
> **TorchScript** 允许我们预编译模型，并且不仅可以在 Python 环境中调用它们，还可以在C++程序和移动设备上调用它们  
> windows通过anaconda或者miniconda安装  
  
>**cuda**：英伟达的类c++语言，被编译并在GPU上并行方式运行  

优点：
- 灵活性、易用性  
- 使用GPU加速计算，比CPU上执行相同的计算速度快50倍  

Theano最早的深度学习框架之一，但已停止开发

工具包：
- torch.nn 提供了通用神经网络层和其他架构组件（激活、损失函数）
- torch.optim 优化器
- torch.Tensor 张量  
图片由RGB通道、空间图像维度（高度、宽度）组成

### Torch Hub机制
- 可以在Github上发布模型
- 无论是否预先训练过权重，都可以通过 PyTorch 可以理解的接口将其公开发布    
- 使用 Torch Hub 通过适当的 hubconf.py 文件从其他任何项目加载模型和权重是一种标准化操作方法  
从Github上加自预训练模型：
> 1. 找到hubconf.py  
```
dependencies = ['torch', 'math'] #代码所依赖的可选模型块列表

def some_entry_fn(*args, **kwargs): 
 model = build_some_model(*args, **kwargs) 
 return model 

def another_entry_fn(*args, **kwargs): 
 model = build_another_model(*args, **kwargs) 
 return model 
#作为存储库入口点向用户暴露一个或多个函数。这些函数应该根据模型初始化参数并将其返回
```
> 2.pytorch/vision 主分支的快照及其权重下载到本地目录  
> 3.默认下载到本地的 torch/hub 目录下  
> 4.然后运行 resnet18 入口点函数，该函数返回实例化的模型  
```
import torch
from torch import hub

resnet18_model = hub.load('pytorch/vision:master','resnet18',pretrained=True)
#GitHub 存储库的名称和分支、函数名称、关键参数)
```

### HDF5 格式和 h5py 库
- HDF5 是一种可移植的、被广泛支持的格式，用于将序列化的多维数组组织在一个嵌套的键值对字典中
- Python 通过 h5py 库支持 HDF5，该库接收和返回 NumPy 数组格式的数据  
<code>$ conda install h5py</code>
- 张量转换为np数组，传递给create_dataset()
```
import h5py 
f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'w') 
dset = f.create_dataset('coords',data=points.numpy()) 
f.close() 
```
- 加载数据集的最后两个点：  
```
# In[62]: 
f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'r') 
dset = f['coords'] 
last_points = dset[-2:]
```
- 一旦完成数据加载，就关闭文件。关闭 HDF5 文件会使数据集失效，然后试图访问 dset 会抛出一
个异常。
