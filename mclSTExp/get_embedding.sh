# list=( 'DC5' 'UC1_I' 'UC1_NI' 'UC6_I' 'UC6_NI' 'UC7_I''UC9_I')

# # Loop through each item in the list
# for i in "${list[@]}"; do
    
#     # python test_beta.py --test_model 100-89 --train True --slide_name "$i" 
#     python test_beta.py  --train False  --test_model 100-89 --slide_name "$i"
# done

# 
python get_embedding_beta.py --test_model 64-109