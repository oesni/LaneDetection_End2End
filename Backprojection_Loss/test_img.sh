python inference.py --loss_policy backproject --evaluate --draw_testset --save_freq 100 --weight_init xavier --use_cholesky 0 --split_percentage 0.1 --activation_layer square --pretrained false --pretrain_epochs 25 --skip_epochs 25 --nclasses 4 --mask_percentage 0.20 --order 3 --clas 1 --nepochs 400 --image_dir "../tuSimple/train_processed/images" --gt_dir "../tuSimple/train_processed/ground_truth" --test_dir "../tuSimple/test_processed"
