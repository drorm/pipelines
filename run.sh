OPENAI_API_KEY=`cat ~/.openai`
PIPELINES_DIR="dev_pipelines"
RESET_PIPELINES_DIR=true
PIPELINES_URLS=''
REQUIREMENTS_FILE=''
PIPELINES_REQUIREMENTS_PATH=''
export ANTHROPIC_API_KEY=`cat ~/.anthropic/api_key`


cd pipelines
rm -f *.py
ln -s ../dev/*.py .
cd ..

# Default to development mode with auto-reload enabled
# Can be overridden by setting UVICORN_EXTRA_FLAGS before running
export UVICORN_EXTRA_FLAGS="--reload"

./start.sh
