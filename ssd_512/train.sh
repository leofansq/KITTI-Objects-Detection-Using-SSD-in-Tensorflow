#!/usr/bin/sh
DATASET_DIR=tfrecords_VOC_KITTI
TRAIN_DIR=logs/finetune_kitti_VOC_512/
CHECKPOINT_PATH=./checkpoints/ssd_512_vgg.ckpt
MODEL_NAME=ssd_512_vgg

python3 train_ssd_network.py --train_dir=${TRAIN_DIR} --dataset_dir=${DATASET_DIR} --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=${MODEL_NAME} --checkpoint_path=${CHECKPOINT_PATH} --save_summaries_secs=600 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --batch_size=12 --gpu_memory_fraction=0.9 --checkpoint_exclude_scopes=ssd_512_vgg/conv6,ssd_512_vgg/conv7,ssd_512_vgg/block8,ssd_512_vgg/block9,ssd_512_vgg/block10,ssd_512_vgg/block11,ssd_512_vgg/block12,ssd_512_vgg/block4_box,ssd_512_vgg/block7_box,ssd_512_vgg/block8_box,ssd_512_vgg/block9_box,ssd_512_vgg/block10_box,ssd_512_vgg/block11_box,ssd_512_vgg/block12_box
