# pip install --upgrade crunch-cli
# git clone https://github.com/CuzzImBatman/Crunch_1.git
# cd ./Crunch_1/
pip install -r requirements.txt
# pip install einops
pip install spatialdata
pip uninstall -y torch

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# crunch setup --notebook --size large broad-1 test --token 2BwKdHlYLLb3FKFYqvyXG9bgZMIonFH52Iu6XQDkIHIb7SvUXWdn6ipSgypRNrqb

