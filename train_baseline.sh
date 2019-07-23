GPU=1,2,3,4
CUDA_VISIBLE_DEVICES=${GPU} \
python train.py \
	--trainRoot /data/home/zjw/dataset/AI+_Chinese_Scene_Text/lmdb/ \
	--workers 2 \
	--batchSize 64 \
	--nepoch 100 \
	--lr 0.1 \
	--expr_dir output/baselineep100_lr0.1_rlop/ \
	--displayInterval 100 \
	--adadelta  \
	--alphabet ./CST_alphabet.txt \
	--random_sample
	--adam