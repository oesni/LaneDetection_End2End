#! /usr/bin/fish
conda activate lane
python main.py --loss_policy backproject --save_freq 100 --weight_init xavier --use_cholesky 0 --split_percentage 0.1 --activation_layer square --pretrained false --pretrain_epochs 25 --skip_epochs 25 --nclasses 4 --mask_percentage 0.20 --order 3 --clas 1 --nepochs 200 --num_train 2787 --image_dir "./culane/train_tusimple/images" --gt_dir "./culane/train_tusimple/ground_truth" --test_dir "./culane/test_tusimple"
