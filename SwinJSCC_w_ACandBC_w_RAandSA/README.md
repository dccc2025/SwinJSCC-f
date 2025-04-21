# SwinJSCC: Taming Swin Transformer for Joint Source-Channel Coding

这个工作是基于这个论文进行修改的"[SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding](https://arxiv.org/abs/2308.09361)".

## Introduction

这个工作是我的本科生毕业设计，主要是对Swin Transformer进行改进，提出了SwinJSCC-f模型。在CIFAR10数据集上进行了大量的实验，并且在武汉大学的MVCD数据集上进行了测试，验证了我提出的模型在图像传输中的有效性。SwinJSCC-f模型在SwinJSCC的基础上完全修改了SNR自适应网络和Rate自适应网络，并且按照Swin Transformer对整体的代码框架没有动。

## Installation
I implement SwinJSCC-f under python 3.12 and PyTorch 2.6.2.（这个倒无所谓，uv创建隔离环境，然后） 

## Note!!
* 这个代码在自己数据集上测试有一点点问题（需要把data_dir从str改成list）————已经修正
* 整体的代码框架我没动，只是将SwinJSCC使用的Swin Transformer V2块替换了Swin

## Models

我把自己修正后的完整版SwinJSCC-f 模型放在谷歌云盘上了 [Google Drive](https://drive.google.com/drive/folders/1_EouRY4yYvMCtamX2ReBzEd5YBQbyesc?usp=sharing). （自己在snr=10, C=32，即cbr=0.33的情况下训的，lr=1e-5, 训练了120个epoch，非常的无敌！）


### For SwinJSCC_w/_SAandRA model 
*e.g. cbr = 0.0208,0.0416,0.0625,0.0833,0.125, snr = 1,4,7,10,13, metric = PSNR, channel = AWGN

```
e.g.
python main.py --trainset DIV2K --testset kodak --distortion-metric MSE --model SwinJSCC_W/O --channel-type awgn --C 32,64,96,128,192 --multiple-snr 1,4,7,10,13 --model_size base
```

>If you want to train this model, please add '--training'. 


## Citation

这个我就不发文章了，本人转方向了，但是我的idea确实可以涨2个点左右（psnr），大家给个star就行了。
有疑问的话 可以留言 可以电联 13801589515（wechat）| 13801589515@163.com （e-mail）

## Acknowledgement
这篇文章基于SwinJSCC的延申，主干网络的修改参考了Swin Transformer V2的架构设计，SNR自适应网络和Rate自适应网络是自己想的，浅浅改了一下，确实涨点（因为实在觉得原来SwinJSCC的那个SA和RA太怪了）。这些是参考工作 [Swin Transformer](https://github.com/microsoft/Swin-Transformer). [SwinJSCC](https://github.com/semcomm/swinjscc). [Swin Transformer V2](https://github.com/ChristophReich1996/Swin-Transformer-V2/).

## Related links
* 
