# sleep 540

CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size 512 --patch_size 80 \
 --embed_dir 'D:/DATA/Gene_expression/Crunch/preprocessed' \
 --device 'cuda' \
 --demo True \
 --local True \

# python get_embedding.py --test_model 100-199
