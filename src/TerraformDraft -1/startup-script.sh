#!/bin/bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io
# Run Label Studio in a Docker container
sudo docker run -d -p 80:8080 -v mydata:/label-studio/data heartexlabs/label-studio:latest

# replace mydata with the desired volume name or path where you want to store Label Studio data.