# KITTI Objects Detection Using SSD in Tensorflow #

## 文件位置 ##

### 数据集 ###
* VOC格式原始数据集 [VOC_ KITTI](/VOC_KITTI)
* VOC -> TFRecord数据集 [tfrecords_ VOC_KITTI](/tfrecords_VOC_KITTI)

### 训练结果 ###
* VOC转换数据集训练结果 [finetune_ kitti_VOC](/logs/finetune_kitti_VOC)

### 检测图片结果 ###
* 检测图片源文件夹  [test_img](/test_img)
* 检测结果生成图片文件夹  [DetectResults](/DetectResults)

## 数据集转换 ##

### KITTI->VOC ###
* 详见TOOLS-Dataset-2-VOC的README
* 将VOC格式的三个文件夹移至 [VOC_ KITTI](/VOC_KITTI)

### VOC->TFRecord ###
* 在[pascalvoc_common.py](/datasets/pascalvoc_common.py) 48行根据实际修改VOC_LABELS
* 在[pascalvoc_2007.py](/datasets/pascalvoc_2007.py) 80和90行修改SPLITS_TO_SIZES中的train和test的个数，和NUM_CLASSES（实际类别数，不用+1）
在SSD/ 执行

        DATASET_DIR=./VOC_KITTI/
		OUTPUT_DIR=./tfrecords_VOC_KITTI
		python3 tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=${DATASET_DIR} --output_name=voc_2007_test --output_dir=${OUTPUT_DIR} --txt_name=test

 
> **注：**txt_name 和 output_name 需根据实际情况改变，训练数据集--train，测试--test

## Training ##
* 在[train_ssd_network.py](/train_ssd_network.py) 中135行修改类别数（实际类别数+1）
* 在[ssd_vgg_300.py](/nets/ssd_vgg_300.py) 中96行修改类别数（实际类别数+1）
* 运行 [train.sh](/train.sh) ,其内容如下

    	DATASET_DIR=tfrecords_VOC_KITTI
		TRAIN_DIR=logs/finetune_kitti_VOC/
		CHECKPOINT_PATH=./checkpoints/ssd_300_vgg.ckpt
		python3 train_ssd_network.py --train_dir=${TRAIN_DIR} --dataset_dir=${DATASET_DIR} --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=${CHECKPOINT_PATH} --save_summaries_secs=600 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --batch_size=32 --gpu_memory_fraction=0.9 --checkpoint_exclude_scopes =ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box
> 训练命令处加上一行checkpoint_exclude_scopes原因：预训练的checkpoints是用21个类别来训练的，与实际训练类别数不同，加上后舍弃了一些层，只用这个原始结构中的weights来训练

## Evaluation ##
* 在[eval_ssd_network.py](/eval_ssd_network.py) 中66行修改类别数（实际类别数+1）
* 运行 [eval.sh](/eval.sh) ,其内容如下

		DATASET_DIR=tfrecords_VOC_KITTI
		EVAL_DIR=./logs/eval_KITTI_VOC/
		CHECKPOINT_PATH=./logs/finetune_kitti_VOC/model.ckpt-7311
		python3 eval_ssd_network.py --eval_dir=${EVAL_DIR} --dataset_dir=${DATASET_DIR} --dataset_name=pascalvoc_2007 --dataset_split_name=test --model_name=ssd_300_vgg --checkpoint_path=${CHECKPOINT_PATH} --batch_size=1

## Demo ##
![demo](/demo.jpg)

> 上图中的测试图片均来源自网络

* [kitti_demo.py](/notebooks/kitti_demo.py) 42行修改模型地址，68行修改测试图片目录地址，72行修改测试图片目录中图片
* [visualization.py](/notebooks/visualization.py) 最后一行取消注释可实现测试图片显示

* 在./notebooks 运行 

    	python kitti_demo.py

> 测试图片保存在[DetectResults](/DetectResults)文件夹中，图片命名为测试源图片名

## VGG_512 ##
* 将[ssd_512](/ssd_512)文件夹中内容替换原有文件即可

## Reference ##
* [SSD: Single Shot MultiBox Detector](http://arxiv.org/abs/1512.02325)
* [SSD in Caffe](https://github.com/weiliu89/caffe/tree/ssd)
* [SSD in Tensorflow](https://github.com/balancap/SSD-Tensorflow)
