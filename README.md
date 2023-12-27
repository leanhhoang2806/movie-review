ssh hoang2@192.168.1.101 => password: 2910
ssh hoang@192.168.1.102  => password 2910

sudo docker build -t movie-review . && sudo docker run movie-review && sudo docker stop movie-review && sudo docker rm movie-review && dsudo ocker rmi --force movie-review

sudo docker stop worker-0 | true && sudo docker rm worker-0 | true && sudo docker rmi --force my_tensorflow_app && sudo docker build -t my_tensorflow_app . && sudo docker run -a stdout -a stderr --gpus all -p 2222:2222 -e TF_CONFIG='{"cluster": {"worker": ["192.168.1.100:2222", "192.168.1.101:2222"]}, "task": {"type": "worker", "index": 0}}' --name worker-0  my_tensorflow_app > training_log.txt
