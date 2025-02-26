#!/bin/bash

# 启动后端服务
echo "启动后端服务..."
cd causal_survey
python api.py &

# 等待后端服务启动
sleep 2

# 启动前端服务
echo "启动前端服务..."
cd ..
npm run dev 