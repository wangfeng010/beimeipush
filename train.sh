set -eu

cd /root/port_service

cat ./hosts >> /etc/hosts

export PYTHONPATH=`pwd`
/opt/conda/bin/python3.8 utils/download.py
ls -lh data/train
/opt/conda/bin/python3.8 train.py 