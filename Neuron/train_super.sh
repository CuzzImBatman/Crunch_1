# python train_super.py --embed_dir 'E:/DATA/crunch/tmp/preprocessed' \
#                                      --batch_size 5  \
#                                      --epoch 100 \
#                                      --save_dir './model_result_super/80_24_1024_1024' \
#                                     --start_epoch 0 \
#                                      --local True\
#                                     --partial  -1 \
python train_super.py --embed_dir 'E:/DATA/crunch/tmp/projection' \
                                     --batch_size 5  \
                                     --epoch 100 \
                                     --save_dir './model_result_super_all/80_24_256_1024' \
                                    --start_epoch 0 \
                                    --input_dim 256 \
                                     --local True\
                                    --partial  -1 \

                                    #  --demo True \
                                     
                                    #  --nolog1p True
                                    #  --encoder_mode True 
                                    #  --centroid_layer True\
                                    
                                    # 
                                    # --encoder_mode True ''D:/DATA/Gene_expression/Crunch/Register/preprocessed'
                                    
                                    
                                    