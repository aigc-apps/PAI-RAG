
sudo yum install postgresql-devel
sudo yum install libffi-devel
python -m pip install numpy cmake wheel pillow --index-url=http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install torch==2.0.1 -f https://download.pytorch.org/whl/torch_stable.html --index-url=http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
python -m pip install --upgrade -r requirements.txt --index-url=http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install --force --no-deps git+https://github.com/UKPLab/sentence-transformers.git
# mkdir embedding_model && cd embedding_model
# wget https://easyrec.oss-cn-beijing.aliyuncs.com/qa-test/SGPT-125M-weightedmean-nli-bitfit.tar.gz
# tar -xvf SGPT-125M-weightedmean-nli-bitfit.tar.gz && rm -rf SGPT-125M-weightedmean-nli-bitfit.tar.gz
