OPENAI_API_KEY=`cat ~/.openai`
PIPELINES_DIR="dev_pipelines"
PIPELINES_REQUIREMENTS_PATH=false

cp dev_pipelines/*.py pipelines
# Default to development mode with auto-reload enabled
# Can be overridden by setting UVICORN_EXTRA_FLAGS before running
UVICORN_EXTRA_FLAGS=${UVICORN_EXTRA_FLAGS:-"--reload"}

./start.sh
