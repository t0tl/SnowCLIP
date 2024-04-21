FROM python:3.11
RUN apt-get update && apt-get upgrade -y 
WORKDIR /workspaces/SnowCLIP
COPY . .
RUN pip install -r requirements.txt
