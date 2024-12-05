
CUDA_LAUNCH_BLOCKING=1 python train.py --embed_dir 'D:/DATA/Gene_expression/Crunch/preprocessed' \
                                     --batch_size 10  \
                                    --epoch 80 \
                                    --demo True
                                    # --encoder_mode True

                                    