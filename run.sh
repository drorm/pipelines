OPENAI_API_KEY=`cat ~/.openai`
PIPELINES_DIR="dev_pipelines"
RESET_PIPELINES_DIR=true
PIPELINES_URLS=''
REQUIREMENTS_FILE=''
PIPELINES_REQUIREMENTS_PATH=''
export ANTHROPIC_API_KEY=`cat ~/.anthropic/api_key`
export WEATHER_API_KEY=`cat ~/.weather`


cd pipelines
rm -f failed/*
rm -f *.py
ln -s ../dev/weather.py .
ln -s ../dev/compute.py .
cd ..

# Default to development mode with auto-reload enabled
# Can be overridden by setting UVICORN_EXTRA_FLAGS before running

./dev.sh
