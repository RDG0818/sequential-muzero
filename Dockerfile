# Use an official NVIDIA CUDA base image.

# For the best results, match this to the CUDA version from your local PC's `nvidia-smi` output.

# For example, if your local machine uses CUDA 12.2, use a 12.2.x tag.

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04


# Set up the environment

ENV DEBIAN_FRONTEND=noninteractive

# Add build-essential for compiling packages

RUN apt-get update && apt-get install -y \

    python3.10 \

    python3-pip \

    git \

    build-essential \

    && rm -rf /var/lib/apt/lists/*


# Create a working directory

WORKDIR /app


# Copy ONLY the requirements file first. This leverages Docker's layer caching.

COPY requirements.txt .


# --- MODIFICATION: Remove hashes from requirements file to handle cross-platform issues ---

# This command finds all lines with hashes and deletes them before pip runs.

RUN sed -i '/--hash/d' requirements.txt


# Install all Python packages from the modified requirements file.

RUN pip install --no-cache-dir --verbose -r requirements.txt


# Now, copy the rest of your project files into the container.
COPY . .

RUN python3 -c "import jax; print('JAX devices:', jax.devices())"