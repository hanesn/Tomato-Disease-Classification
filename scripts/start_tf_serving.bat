@echo off
setlocal enabledelayedexpansion

REM Navigate to root dir from scripts/
cd /d "%~dp0\.."

REM Load environment variables from .env
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
    set "%%A=%%B"
)

REM Validate required environment variables
if not defined MODEL_DIR (
    echo [ERROR] MODEL_DIR not set in .env
    exit /b 1
)

if not defined TF_SERVING_MOUNT_PATH (
    echo [ERROR] TF_SERVING_MOUNT_PATH not set in .env
    exit /b 1
)

if not defined TF_SERVING_CONFIG_PATH (
    echo [ERROR] TF_SERVING_CONFIG_PATH not set in .env
    exit /b 1
)

set CONTAINER_NAME=tf_serving_container
set ROOT_DIR=%cd%

REM Download the tensorflow serving docker image
docker pull tensorflow/serving

REM Stop and remove any existing container
docker stop %CONTAINER_NAME% >nul 2>&1
docker rm %CONTAINER_NAME% >nul 2>&1

REM Actual docker run command
docker run -d --name %CONTAINER_NAME% -p 8501:8501 ^
  -v "%ROOT_DIR%\%MODEL_DIR%:/%TF_SERVING_MOUNT_PATH%" ^
  -v "%ROOT_DIR%\%TF_SERVING_CONFIG_PATH%:/models.config" ^
  tensorflow/serving --model_config_file=/models.config

REM Report status
echo TensorFlow Serving container started as '%CONTAINER_NAME%'
