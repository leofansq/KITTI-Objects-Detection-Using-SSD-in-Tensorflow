#DATASET_DIR=tfrecords_VOC_KITTI
#EVAL_DIR=./logs/eval_KITTI_VOC/
#CHECKPOINT_PATH=./logs/finetune_kitti_VOC_full/model.ckpt-9822
#python3 eval_ssd_network.py --eval_dir=${EVAL_DIR} --dataset_dir=${DATASET_DIR} --dataset_name=pascalvoc_2007 --dataset_split_name=test --model_name=ssd_512_vgg --checkpoint_path=${CHECKPOINT_PATH} --batch_size=1

DATASET_DIR=tfrecords_VOC_KITTI
EVAL_DIR=./logs/eval_KITTI_VOC/
CHECKPOINT_PATH=./logs/finetune_kitti_VOC/model.ckpt-7311
python3 eval_ssd_network.py --eval_dir=${EVAL_DIR} --dataset_dir=${DATASET_DIR} --dataset_name=pascalvoc_2007 --dataset_split_name=test --model_name=ssd_300_vgg --checkpoint_path=${CHECKPOINT_PATH} --batch_size=1
