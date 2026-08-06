FROM python:3.11-slim

# --- System dependencies -----------------------------------------------
# ffmpeg:        video demuxing + audio extraction (used by ego4d clips,
#                librosa's audioread fallback, and any segmenting you do
#                with moviepy/ffmpeg-python)
# libsndfile1:   required by `soundfile`, which librosa uses for fast WAV I/O
# git:           some pip packages (incl. occasional ego4d deps) install from VCS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp for fast C++ ASR inference
RUN git clone https://github.com/ggerganov/whisper.cpp.git /app/whisper.cpp && \
    cd /app/whisper.cpp && \
    make -j4 && \
    ./models/download-ggml-model.sh medium.en

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Scratch space for in-flight downloads/segments before they're pushed to
# Swift object storage. Mount a Nectar volume here in docker-compose so it
# survives container restarts and isn't limited by the instance's root disk.
RUN mkdir -p /data/scratch
ENV SCRATCH_DIR=/data/scratch

CMD ["python", "-m", "src.pipeline"]
