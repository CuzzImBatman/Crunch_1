# /Crunch_data  /root/Crunch_1/data/

#sleep n
vastai cloud copy --src /root/Crunch_1/mclSTExp/model_result/100 --dst /Crunch_model_save --instance 13669863 --connection 18810 --transfer "Instance To Cloud"
sleep 60
# vastai destroy instance xxxxxxx
