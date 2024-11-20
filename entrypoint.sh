#!/bin/bash

# Configure git credentials at runtime
if [ ! -z "$GITHUB_TOKEN" ]; then
    echo "Configuring git with GITHUB_TOKEN..."
    git config --global url."https://${GITHUB_TOKEN}:x-oauth-basic@github.com/".insteadOf "git://github.com/"
    git config --global url."ssh://git@github.com/".pushInsteadOf "https://github.com/"
fi

# Execute whatever command was passed to docker run
exec "$@"