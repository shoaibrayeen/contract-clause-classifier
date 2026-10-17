#!/bin/sh
# Entry point of the tools image (see Dockerfile). Checks that the project folder is really mounted at the working
# directory set by docker-compose.yml before running the requested command: with Colima or Docker Desktop, a host
# folder that is not shared with the Docker VM is mounted as an empty directory, and every script would then fail
# with a confusing "file not found".
set -e

if [ ! -f pyproject.toml ] || [ ! -d scripts ]; then
    echo "The project folder is not visible inside the container ($PWD is empty)." >&2
    cat >&2 <<'EOF'

Docker runs in a virtual machine that only sees the host folders it shares. Share the folder that contains this
project, then run the command again:

  Colima:          add it under `mounts:` in ~/.colima/default/colima.yaml (writable: true), then `colima restart`
  Docker Desktop:  Settings > Resources > File sharing

Or run this step without Docker:  make <target> RUNNER=local
EOF
    exit 2
fi

exec "$@"
