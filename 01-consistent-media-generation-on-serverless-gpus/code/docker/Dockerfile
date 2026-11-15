# Lessons 03 and 04: the Santa cards worker.
# Sana + Wan2.2 I2V + Kokoro + ffmpeg in one image, for one GPU Job.
# CUDA runtime base pinned by tag and digest; weights, secrets and outputs stay outside.
# Re-pin the digest on purpose (e.g. for security fixes).
# Build from the module folder, because the image needs files from code/ and from assets/:
#   cd 01-consistent-media-generation-on-serverless-gpus
#   docker buildx build --platform linux/amd64 -f code/docker/Dockerfile .
FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/cache/huggingface
# espeak-ng: Kokoro's fallback for words its dictionary doesn't know.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng && rm -rf /var/lib/apt/lists/*
RUN pip install diffusers==0.39.0 transformers==5.13.1 accelerate==1.14.0 sentencepiece==0.2.1 protobuf==6.33.0
# ftfy: Wan prompt cleaning. imageio-ffmpeg: MP4 export and the mix step.
# en_core_web_sm: Kokoro's English pipeline. Installed here so a Job does not pip-install it at runtime.
RUN pip install kokoro==0.9.4 ftfy==6.3.1 imageio==2.38.1 imageio-ffmpeg==0.8.1 pillow==12.2.0 pydantic==2.14.3 typer==0.29.1 rich==15.0.0 \
    && pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
# Same layout as the repo: the script in /app reads the default kids list, fonts and music from
# ../assets. /data is where the Job mounts your bucket: inputs you upload, outputs it writes.
WORKDIR /app
COPY code/common.py code/models.py code/batch_inference.py /app/
COPY assets/kids.csv assets/4_jingle.mp3 /assets/
COPY assets/fonts /assets/fonts
RUN mkdir -p /cache /data
CMD ["python", "/app/batch_inference.py", "--n", "1", "--out", "/data/run"]
