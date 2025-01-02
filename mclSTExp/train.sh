# sleep 540

CUDA_LAUNCH_BLOCKING=1 python train.py --batch_size 3200 --path_save './model_result_centroid/24' \
 --embed_dir 'E:/DATA/crunch/tmp/preprocessed' \
 --cluster_dir 'E:/DATA/crunch/tmp/cluster' \
 --device 'cuda' \
 --local True \
 --centroid True \
  --demo True \
#  --centroid True\

# python get_embedding.py --test_model 100-199
