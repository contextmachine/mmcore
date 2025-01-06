# syntax=docker/dockerfile:1
#        |     No ENTRYPOINT      |	 ENTRYPOINT exec_entry p1_entry	 | ENTRYPOINT [“exec_entry”, “p1_entry”]          |
# -------+------------------------+----------------------------------+------------------------------------------------+
# No CMD | error, not allowed	  | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry                            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | [“exec_cmd”, “p1_cmd”] | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry exec_cmd p1_cmd            |
# -------+------------------------+----------------------------------+------------------------------------------------+
# CMD    | exec_cmd p1_cmd	      | /bin/sh -c exec_entry p1_entry   | exec_entry p1_entry /bin/sh -c exec_cmd p1_cmd |

# Stage 1: Build
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mmcore

COPY . .

RUN pip wheel install --user --no-cache-dir .

# Stage 2: Final
FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/contextmachine/mmcore
LABEL org.opencontainers.image.description="mmcore, the modern cad/cam engine"
LABEL autor="Andrew Astakhov <sthv.developer@gmail.com>"
# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local


EXPOSE 8000
# Define the entrypoint
ENTRYPOINT ["/bin/bash"]
