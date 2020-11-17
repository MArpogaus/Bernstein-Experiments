FROM tensorflow/tensorflow:1.12.0-gpu-py3

WORKDIR /app

ADD . /app

RUN pip install git+https://github.com/MArpogaus/TensorFlow-Probability-Bernstein-Polynomial-Bijector.git

RUN pip install git+https://github.com/MArpogaus/tensorflow-experiments.git

RUN pip install .

RUN pip install tensorboard jupyter

ENTRYPOINT "entrypint.sh"

CMD ["jupyter-notebook;", "tensorboard --logdir ./logs"]