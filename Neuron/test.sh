
# CUDA_LAUNCH_BLOCKING=1 python test.py --embed_dir 'E:/DATA/crunch/tmp/preprocessed' \
# --batch_size 50  \
# --start_epoch 99 --device 'cuda' \
# --save_dir './model_result/80_24_1024_1024' \
# --demo True \
# --encoder_mode True \
CUDA_LAUNCH_BLOCKING=1 python test.py --embed_dir 'E:/DATA/crunch/tmp/projection' \
--batch_size 50  \
--start_epoch 11 --device 'cuda' \
--save_dir './model_result/80_24_256_1024' --input_dim 256 \