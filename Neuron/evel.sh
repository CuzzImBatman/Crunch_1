
CUDA_LAUNCH_BLOCKING=1 python evel.py --embed_dir 'D:/DATA/Gene_expression/Crunch/preprocessed' \
 --batch_size 50  \
 --save_dir './model_result/80_1024_1024' \
  --epochs 91 --device 'cuda' \
  --demo True
# --encoder_mode True \
