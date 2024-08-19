FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip python3.11 python3.11-dev python3.11-distutils curl \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

COPY . /app

RUN python3 -m pip install poetry

RUN poetry config virtualenvs.create false

RUN poetry install --no-root

CMD ["poetry", "run", "python", "-m", "adalflow"]