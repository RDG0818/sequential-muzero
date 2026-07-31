# GPU-enabled image for sequential-muzero (SMAX 3m MuZero training).
# Requires the NVIDIA Container Toolkit on the host: docker run --gpus all ...
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# build-essential: cpprb (replay buffer) compiles a C extension at install time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .
# requirements.txt pulls CPU jax first; jax[cuda12] afterward replaces jaxlib
# with the CUDA build. Same two-step install the user hit by hand on bare metal.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --upgrade "jax[cuda12]"

COPY . .

ENTRYPOINT ["python", "train.py"]
CMD ["model=smax", "mcts=joint"]
