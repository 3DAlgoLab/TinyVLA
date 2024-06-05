# docker build -t experiments .
sudo docker run --gpus all -it --rm -v $(pwd):/app experiments
