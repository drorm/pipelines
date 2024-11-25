cd /share/python2/pipelines
bash run.sh &

PID=$!

echo sleeping 6

sleep 6
#

# Kill the background process
kill $PID
killall uvicorn
