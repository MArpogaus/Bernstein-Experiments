FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt update; \
    apt install -y git; \
    pip install --upgrade pip;

RUN pip install tensorboard

RUN pip install git+https://github.com/MArpogaus/TensorFlow-Probability-Bernstein-Polynomial-Bijector.git

RUN pip install git+https://github.com/MArpogaus/tensorflow-experiments.git@dev

WORKDIR /app

ADD . /app

RUN pip install .

ENTRYPOINT ["./entrypoint.sh"]
