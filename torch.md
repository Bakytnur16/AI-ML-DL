# 过程
![image](https://user-images.githubusercontent.com/64322636/219554369-87b18509-bcf0-4d16-9331-60e370b59ba7.png)

1.数据处理：  
数据样本转换为张量  
将多个张量组成批样本  

> 预训练网络是已经在数据集上训练过的模型。这类网络通常可以在加载网络参数后立即产生有用的结果
> 通过了解如何使用预训练模型，我们可以将神经网络集成到一个项目中，而不需要对其进行设计和训练

## 图像识别  
![image](https://user-images.githubusercontent.com/64322636/219554377-04062976-1ac4-4341-8325-26fa50ef1556.png)
在深度学习中，在新数据上运行训练过的模型的过程被称为推理（inference）。为了进行推理，我们需要将网络置于 eval 模式  
> 如果忘记加eval：那么一些预先训练过的模型，如批归一化（Batch Normalization）和Dropout 将不会产生有意义的答案，这仅仅是因为它们内部工作的方式。现在 eval 设置好了，我们准备进行推理    

AlexNet，它是在图像识别方面早期具有突破性的网络之一。AlexNet 和 ResNet【2015】 是 2 个深度卷积神经网络，它们的发布为图像识别设定了新的基准。

![image](https://user-images.githubusercontent.com/64322636/219554569-c857b1e8-4fe1-42e0-b678-e69826076e23.png)


## 自然语言处理
#### 为图像配上字幕（说明）
![image](https://user-images.githubusercontent.com/64322636/219554772-30eca3cc-539c-4f23-83a4-52249502933f.png)

- **卷积神经网络**负责识别图片上的所有信息：‘粉红色气球’、‘人’、‘头发’ -> 它学习生成场景的“描述性”数字表征
- **循环神经网络**负责产生连贯的句子：一个拿着粉红色气球的有头发的人。 -> 它通过将这些描述性的数字放在一起产生一个连贯的句子
> 循环神经网络: 它在随后的正向传播中生成输出，即单个单词，其中每个正向传播的输入包括前一个正向传播的输出。这将使下一个单词对前面生成的单词产生依赖性，就像我们在处理句子或处理序列时所期望的那样

![image](https://user-images.githubusercontent.com/64322636/219562394-07ad0d5f-e386-4932-bd44-db40abf8c3b3.png)

# 张量 - 多维数组
![image](https://user-images.githubusercontent.com/64322636/219562667-6d2614a0-d57b-44ee-9ba8-0dcdd3ae1349.png)
- NumPy 是最受欢迎的多维数组库

python列表与张量的区别：
![image](https://user-images.githubusercontent.com/64322636/219563994-e864f166-6f31-4eb5-8929-4dc0ea299c86.png)
