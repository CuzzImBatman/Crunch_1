
CUDA_LAUNCH_BLOCKING=1 python train.py --embed_dir 'D:/DATA/Gene_expression/Crunch/preprocessed' \
                                     --batch_size 73  \
                                     --epoch 100 \
                                     --save_dir './model_result/80_1024_1024' \
                                    --start_epoch 0 \
                                     --local True\
                                    #  --demo True \
                                    #  --encoder_mode True 
                                    #  --centroid_layer True\
                                    
                                    # 
                                    # --encoder_mode True
                                    
                                    
                                    