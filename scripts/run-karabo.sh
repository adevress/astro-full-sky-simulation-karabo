#!/bin/bash

mkdir -p userdir
docker run --rm -v $PWD:/workspace/userdir -p 8888:8888 ghcr.io/i4ds/karabo-pipeline:latest bash -c 'jupyter lab --ip 0.0.0.0 --no-browser --allow-root --port=8888'

