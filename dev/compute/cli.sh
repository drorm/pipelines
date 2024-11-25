#!/bin/bash

export ANTHROPIC_API_KEY=`cat ~/.anthropic/api_key`

cd /share/python2/pipelines/dev

PYTHONPATH=/share/python2/pipelines/dev python3 -m new.cli "$@"
