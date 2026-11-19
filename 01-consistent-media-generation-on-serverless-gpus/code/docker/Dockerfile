# Lessons 03 and 04: the Santa cards worker.
# Sana + Wan2.2 I2V + Kokoro + ffmpeg in one image, for one GPU Job.
# CUDA runtime base pinned to a version tag; weights, secrets and outputs stay outside.
# Build from the module folder, because the image needs files from code/ and from assets/:
#   cd 01-consistent-media-generation-on-serverless-gpus
#   docker buildx build --platform linux/amd64 -f code/docker/Dockerfile -t santa-worker:local .
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/cache/huggingface

WORKDIR /app

# espeak-ng: Kokoro's fallback for words its dictionary doesn't know.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/*
COPY code/docker/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Same layout as the repo: the script in /app reads the default kids list, fonts and music from
# ../assets. /data is where the Job mounts your bucket: inputs you upload, outputs it writes.
RUN mkdir -p /cache /data

COPY code/common.py code/models.py code/batch_inference.py /app/
COPY assets/kids.csv assets/4_jingle.mp3 /assets/
COPY assets/fonts /assets/fonts

ENTRYPOINT ["python", "/app/batch_inference.py"]
CMD ["--n", "1", "--out", "/data/run"]
