###  使用Tensorflow deeplabv3+ 训练自己的数据集

参考:https://blog.csdn.net/heiheiya/article/details/88535576

##### 1.DeepLabV3与DeepLabV3+

>DeepLabV3  是另一种无需加设系数的多尺度处理方法。  
>
>这个模型十分轻量级。我们再次从一个特征提取前端开始，从第四次下采样后的特征入手处理。现在的分辨率已经很低（比输入图片小了16倍）所以我们就从这里入手就很好！不太好的一点是，在这样一种低分辨率的情况下，由于像素的低准确度，很难得到很好的定位。
>
>这就是体现 DeepLabV3 的突出贡献之处了，对多孔卷积的巧妙运用。普通的卷积只能处理局部信息，因为权值总是一个挨着一个。例如，在一个标准的3*3卷积中，两个权重值之间的距离只有一个步长/像素。
>
>有了多孔卷积，我们可以直接增加卷积权重值之间的空间，而实际上在操作中不增加权重值的总数。所以我们仍然只有3*3也就是9个为参数总量，我们只是把它们分布得更开了。我们把每个权重值间的距离称作扩张率。下面的模型图解很好的阐释了这个思想。
>
>当我们使用一种低扩张率时，我们会得到非常局部/低尺度的信息。当我们采用高扩张率时，我们会处理到更多全局/高尺度的信息。因此 DeepLabV3 模型融合了不同扩张率的多孔卷积来获取多尺度信息。
>
>![12](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/12.png)



>DeepLabV3+ 模型就像它名字所说的那样，是 DeepLabV3 的一个快速延伸，借用了它之前工作的一些优势。如我们之前看到的那样，如果我们仅仅在最后利用双线性差值升尺度的话就会遇到潜在的瓶颈。事实上，原来的 DeepLabV3 在最后把尺度放大了16倍！ 
>
>为了处理这件事， DeepLabV3+ 在 DeepLabV3 上加入了中间的解码模块，通过 DeepLabV3 处理后，首先特征图会被放大4倍。之后它们会和前端特征提取的原特征图一起处理，之后再放大4倍。这减轻了后端网络的负担，并提供了一条从前端特征提取到网络后面部分的捷径。
>
>![13](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/13.png)



##### 2.制作语义分割数据集

下载30张道路和车辆的图片

![20](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/20.png)

labelme的使用

>labelme是麻省理工（MIT）的计算机科学和人工智能实验室（CSAIL）研发的图像标注工具，人们可以使用该工具创建定制化标注任务或执行图像标注，项目源代码已经开源。

项目开源地址：https://github.com/CSAILVision/LabelMeAnnotationTool

labelMe项目地址：http://labelme.csail.mit.edu/Release3.0/

安装

```
sudo apt-get install python-pyqt5
sudo pip install labelme
```

使用

![15](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/15.png)

> 报错1：路径文件夹名称不要包含中文



初始界面

<img src="https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/18.png" alt="18" style="zoom: 50%;" />

OpenDir可以直接批量处理文件夹内的图片

使用CreatePolygons进行多边形框选，框选目标后可以设置标签名称

<img src="https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/19.png" alt="19" style="zoom:50%;" />

完成后会生成json文件

![4](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/4.png)

labelme下面的cli文件夹下有一个`json_to_dataset.py`，执行该文件，解析json文件

![5](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/5.png)

使用脚本对文件批量处理

```
num=31
for ((i=1;i<num;i++))
do
  python json_to_dataset.py /home/hp/car_label/$i.json -o /home/hp/car_label/output/$i
done
```

处理后

![6](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/6.png)每个文件夹对应一张图片，文件夹内有五个文件

![7](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/7.png)

接下来为标注出来的label.png进行着色

写一个python程序，命名为`convert.py`

```python
import PIL.Image
import numpy as np
from skimage import io,data,color
import matplotlib.pyplot as plt
 
num=31
for i in range(num):    
    img=PIL.Image.open("/home/hp/car_label/output/%d/label.png"%i)
    img=np.array(img)
    dst=color.label2rgb(img,bg_label=0,bg_color=(0,0,0))
    io.imsave("/home/hp/car_label/output/dstlabel/%d.png"%i,dst)
```

着色后的结果

![8](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/8.png)

刚刚着色之后的图是24位png图，deeplab中的标注图片需要是灰度图

原博主使用matlab进行转换

```matlab
dirs=dir('car_label/output/*.png');
for n=1:numel(dirs)
    strname=strcat('car_label/output/',dirs(n).name);
    img=imread(strname);
    [x,map]=rgb2ind(img,256);
    newname=strcat('car_label/output/dstlabel/',dirs(n).name);
    imwrite(x,newname);
end
```

这是生成的结果

![9](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/9.png)

因为图像的像素值其实是像素的类别，是非常小的，看起来是一片黑的

![14](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/14.png)

使用matlab可以看出区别

至此，我们的原始数据集基本制作完成



##### 3.使用 DeepLabv3+ 训练数据集

编写python程序 `clist.py` 生成训练集的 train.txt 

```python
import os
import random
 
trainfilepath = 'train'
txtsavepath = 'txt'
train_file = os.listdir(trainfilepath)
 
num=len(train_file)
list = range(num)
 
train = random.sample(list, num)  
 
os.chdir(txtsavepath)   
 
ftrain = open('train.txt', 'w')  
 
for i in list :
  name =train_file[i][:-4] + '\n'
  ftrain.write(name)
ftrain.close()
```

按照同样的方法生成 val.txt

本次使用22张图片作为训练集，5张图片作为验证集，3张图片作为测试集

生成的train.txt如图所示

<img src="https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/17.png" alt="17" style="zoom:50%;" />

下面生成tfrecord文件

>TFRecord 是什么？
>
>TFRecord 是谷歌推荐的一种二进制文件格式，理论上它可以保存任何格式的信息
>
>下面是Tensorflow 的官网给出的文档结构，整个文件由文件长度信息、长度校验码、数据、数据校验码组成
>
>```python
>uint64 length
>uint32 masked_crc32_of_length
>byte   data[length]
>uint32 masked_crc32_of_data
>```
>
>但对于我们普通开发者而言，我们并不需要关心这些，Tensorflow 提供了丰富的 API 可以帮助我们轻松读写 TFRecord 文件



>优点：
>
>1、它特别适应于 Tensorflow ，或者说它就是为 Tensorflow 量身打造的。  
>2、因为 Tensorflow开发者众多，统一训练时数据的文件格式是一件很有意义的事情。也有助于降低学习成本和迁移成本。

使用`models_master/research/deeplab/datasets/build_voc2012_data.py`来生成

按照要求整理文件夹

```python
  + pascal_voc_seg
    - build_data.py
    - build_voc2012_data.py (current working directory).
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
        + ImageSets
          + Segmentation
    + tfrecord

Image folder:
  ./VOCdevkit/VOC2012/JPEGImages

Semantic segmentation annotations:
  ./VOCdevkit/VOC2012/SegmentationClass

list folder:
  ./VOCdevkit/VOC2012/ImageSets/Segmentation
```

由于图片数量比较少，所以将81行的_NUM_SHARDS = 2，默认是4

生成的tfrecord如下图所示

![16](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/16.png)

使用现有模型训练时，首先要注册自己的数据集

在`models_master/research/deeplab/datasets/segmentation_dataset.py`中110行的地方添加如下内容

```python
_SEGTEST_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 22,  # num of samples in images/training
        'val': 5,  # num of samples in images/validation
    },
    num_classes=4,
    ignore_label=255,
)
```

`ignore_label`是不参与计算loss的，在mask中将`ignore_label`的灰度值标记为`255`

在120行把自己的数据集注册进去

```python
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'segtest': _SEGTEST_INFORMATION
}
```

最后进行训练

```python
WORK_DIR="./"
DATASET_DIR="./datasets/"

 
# Set up the working directories.
SEG_FOLDER="mysegtest"
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${DATASET_DIR}/${SEG_FOLDER}/init_models"
TRAIN_LOGDIR="${DATASET_DIR}/${SEG_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${DATASET_DIR}/${SEG_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${DATASET_DIR}/${SEG_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${DATASET_DIR}/${SEG_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"
 
 
SEG_DATASET="${DATASET_DIR}/tfrecord"
 
# Train 10 iterations.
NUM_ITERATIONS=500
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size=513 \
  --train_crop_size=513 \
  --train_batch_size=1 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --dataset="segtest" \
  --tf_initial_checkpoint="./backbone/deeplabv3_cityscapes_train/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${SEG_DATASET}" \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True
```

原博主`train_batch_siaze`设置为4，运行时发现超出内存容量，改为1后正常运行

运行时每10次输出一个loss值

![1](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/1.png)

![2](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/2.png)

共运行500次，平均每次约9.1s，共耗时一个半小时左右

loss的变化趋势

![11](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/11.png)

训练之后进行验证

```python
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=514 \
  --eval_crop_size=794 \
  --dataset="segtest" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${SEG_DATASET}" \
  --max_number_of_evaluations=1
```

![3](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/3.png)

图片尺寸可能会报错

根据自己的数据集中最大的图片尺寸更改`eval_crop_size`即可

最后进行可视化

```python
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size=514 \
  --vis_crop_size=794 \
  --dataset="segtest" \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${SEG_DATASET}" \
  --max_number_of_iterations=1
```

在`mysegtest/exp/train_on_trainval_set/vis/segmentation_results`可以找到可视化之后的结果

![10](https://github.com/HP2019/Semantic-Segmentation/blob/master/20191013%E7%BB%84%E4%BC%9A/%E5%9B%BE%E7%89%87/10.png)

根据结果进行模型修正等工作
