# How to use PyCharm & Docker with GPU 

1. Run `nvidia-smi` to check CUDA and Driver version
2. Pull the base image from DockerHub
3. Run `docker build` to build the image
4. Run  
`nvidia-docker run -d -p 10022:22 -p 8000:8000 -it  deep_rl:v1.7 /bin/bash
`  
to build container from the Image... where `deep_rl:v1.7` is the name of my image.  
Note that need to use `nvidia-docker`, otherwise, container cannot find gpu.
`-p` is to set the mapping from ports of the container to machine

5. Set ssh for the container with  
`
apt-get update
apt-get install openssh-server  
apt-get install vim  
`

6. `vim /etc/ssh/sshd_config`, change `PermitRootLogin prohibit-password` to `PermitRootLogin prohibit-password`
and uncomment it
7. `passwd root`
8. `service ssh restart`

Then, you need configure Pycharm to remote-connect to the container. 