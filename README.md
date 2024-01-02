ssh hoang2@192.168.1.101 => password: 2910
ssh hoang@192.168.1.102  => password 2910

sudo docker stop movie-review | true && sudo docker rm movie-review | true && sudo docker rmi --force movie-review && sudo docker build -t movie-review . && sudo docker run movie-review --gpus all

running on 101 machine
git pull && sudo docker stop worker-0 | true && sudo docker rm worker-0 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run -a stdout -a stderr --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.101:2222", "192.168.1.102:2222"]}, "task": {"type": "worker", "index": 0}}' --name worker-0  my_tensorflow_app


Running on 102 machine
git pull && sudo docker stop worker-1 | true && sudo docker rm worker-1 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.101:2222", "192.168.1.102:2222"]}, "task": {"type": "worker", "index": 1}}' --name worker-1  my_tensorflow_app


The NVIDIA graphic card needs to go with a specific tensorflow gpu image. For example,
card GTX 1660 ti must have tensorflow image 2.3.0 . Do NOT use latest tag !!!
The image will come with a specified python version. Make sure to create a virtualenv with this 
specific version. Install a specific version of python
then find the binary path
$which python3.8
generate virtual env with the path
$virtualenv -p /usr/local/bin/python3.8 myenv

# Docker clean up
sudo docker system prune


instruction on installing nvidia container toolkit 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html