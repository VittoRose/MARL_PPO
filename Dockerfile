FROM python:3.12.3-slim 

RUN pip install torch gymnasium tensorboard 

CMD [ "bash" ]