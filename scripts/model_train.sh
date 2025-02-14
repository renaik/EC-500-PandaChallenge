python main.py \
--n_class 6 \
--train_images 'data/train_images' \
--val_images 'data/val_images' \
--train_csv 'data/train.csv' \
--val_csv 'data/val.csv' \
--train_graphs 'data/train_graphs/simclr_files' \
--val_graphs 'data/val_graphs/simclr_files' \
--model_path 'saved_models/' \
--log_path 'runs/' \
--task_name "GraphCAM" \
--batch_size 32 \
--lr 0.001 \
--n_epoch 200 \
--weight_decay 0.01 \
--gamma 0.1 \
--log_interval_local 6 \