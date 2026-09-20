# Refinery — raw text in, training-grade corpus out.
#
# This is a CLI tool, not a server. The image's default command runs the
# test suite once (a built-in self-check that the image works), then prints
# the CLI usage.
#
#   build:  docker build -t refinery .
#   run:    docker run --rm refinery
#   refine: docker run --rm -v "$PWD:/data" refinery \
#             python -m refinery.cli --input /data/my-corpus \
#               --out /data/corpus.clean.txt --report /data/report.md
FROM python:3.12-slim

WORKDIR /app

# The package itself is pure standard library — pytest is only needed for
# the self-check (tests) the default command runs.
RUN pip install --no-cache-dir pytest

COPY refinery/ refinery/
COPY tests/ tests/
COPY README.md .

# Self-check first, then usage: a failing suite fails the container.
CMD ["sh", "-c", "python -m pytest tests/ -q && echo '— self-check passed —' && echo && python -m refinery.cli --help"]
