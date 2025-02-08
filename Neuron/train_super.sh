# python train_super.py --embed_dir 'E:/DATA/crunch/tmp/preprocessed' \
#                                      --batch_size 5  \
#                                      --epoch 100 \
#                                      --save_dir './model_result_super/80_24_1024_1024' \
#                                     --start_epoch 0 \
#                                      --local True\
#                                     --partial  -1 \
CUDA_LAUNCH_BLOCKING=1 python train_super.py --embed_dir 'E:/DATA/crunch/tmp/projection_super' \
                    --cluster_path 'E:/DATA/crunch/tmp/cluster'\
                                     --batch_size 6 \
                                     --epoch 100 \
                                     --save_dir './model_result_super_all_testing/80_24_256_1024_no1p' \
                                    --start_epoch 0 \
                                    --input_dim 256 \
                                    --ratio_sample 1 \
                                     --local True\
                                    --partial  -1 \
                                     --demo True \
                                     --nolog1p True
                                    #  --encoder_mode True 
                                    #  --centroid_layer True\
                                    
                                    # 
                                    # --encoder_mode True ''D:/DATA/Gene_expression/Crunch/Register/preprocessed'
                                    
                            #  D:/DATA/Gene_Expression/crunch/patches       
                                    