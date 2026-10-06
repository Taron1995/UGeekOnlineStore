FROM python:3.11-slim

# FFmpeg: сборка без --enable-gpl (LGPL). Клипы режутся с "-c copy", кодеки GPL не нужны.
# Проверьте актуальность URL; можно подставить свою LGPL-сборку.
ARG FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl.tar.xz
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils libgl1 libglib2.0-0 \
    && curl -fsSL "$FFMPEG_URL" | tar -xJ --strip-components=2 -C /usr/local/bin --wildcards '*/bin/ffmpeg' \
    && rm -rf /var/lib/apt/lists/*
# Сборка падает, если FFmpeg оказался GPL-вариантом
RUN ffmpeg -version | head -1 && ! ffmpeg -version | grep -q -- '--enable-gpl'

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "monitor.run", "worker"]
