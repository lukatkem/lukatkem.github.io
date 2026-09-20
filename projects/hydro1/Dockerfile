# Hydro-1 — a GPT trained from zero: serving image for the API + playground UI.
#
#   build:  docker build -t hydro1 .
#   run:    docker run --rm -p 8001:8001 -v "$PWD/checkpoints:/app/checkpoints" hydro1
#
# Trained weights live in checkpoints/ (gitignored, .dockerignored) and are
# mounted at runtime — the image itself ships only code, so it stays small.
# Mounting is optional: without a checkpoint the server still boots and the
# UI reports that no model is loaded.
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch wheels — a fraction of the default CUDA download, and all
# this project ever needs.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Package + playground UI (checkpoints/ and corpus/*.txt excluded via .dockerignore)
COPY hydro1/ hydro1/
COPY web/ web/

EXPOSE 8001

CMD ["python", "-m", "hydro1.server"]
