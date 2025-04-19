#!/bin/bash

# Start backend service
echo "Starting backend service..."
cd causal_survey
python api.py &

# Wait for backend service to start
sleep 2

# Start frontend service
echo "Starting frontend service..."
cd ..
npm run dev 