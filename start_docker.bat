@echo off
REM ============================================================
REM  Traffic Monitoring — Full Pipeline Launcher
REM ============================================================

REM Build individual images (still useful for standalone testing)
echo [1/2] Building traffic-split image...
docker build -t traffic-split -f Split/Dockerfile .

echo [2/2] Building traffic-tracking image...
docker build -t traffic-tracking -f Tracking/Dockerfile .

echo.
echo Done building individual images.
echo.
echo To launch the full pipeline (Kafka + Spark + Split + Tracking):
echo   docker compose up
echo.
echo To scale the Tracking service horizontally (e.g. 4 workers):
echo   docker compose up --scale tracking=4
echo.
echo Spark Web UI available at: http://localhost:8080
echo Kafka broker available at: localhost:9092
