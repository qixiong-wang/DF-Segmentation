# RSSegmentation
Remote Image Segmentation  
模型主要基于DA-net的结构，采用resnet101作为backbone，卷积后分辨率变为1/8，经过DA的模块得到新特征，上采样过程中额外设计了Unet结构与前面层的特征进行融合。  
在数据层面上，加入了随机旋转、镜像、对比度增强、亮度增强等增强方案用于训练。  
模型链接: https://pan.baidu.com/s/1oMmT_kcVbNs5SKzqUdTl4Q  密码: bmnd
