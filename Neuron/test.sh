
CUDA_LAUNCH_BLOCKING=1 python test.py --embed_dir 'D:/DATA/Gene_expression/Crunch/preprocessed' \
--batch_size 512  \
--encoder_mode True \
--epoch 80 --device 'cpu'