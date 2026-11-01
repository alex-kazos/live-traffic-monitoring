@echo off
REM ============================================================
REM  Traffic Monitoring - Pipeline Launcher
REM
REM  Builds and starts every service in docker-compose.yml:
REM    kafka, kafka-init, tracking-papagou, tracking-roupaki,
REM    spark-analytics, spark-alerts, dashboard
REM ============================================================

REM Run from the project root so docker compose finds docker-compose.yml
cd /d "%~dp0"

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Start Docker Desktop and try again.
    exit /b 1
)

echo Building images and starting the pipeline...
docker compose up -d --build --remove-orphans
if errorlevel 1 (
    echo.
    echo docker compose failed - see the output above.
    exit /b 1
)

echo.
docker compose ps
echo.
echo Dashboard:     http://localhost:8090
echo Kafka broker:  localhost:9092
echo.
echo Live analytics (Plotly, runs outside Docker):
echo                pip install -r LocalDashboard\requirements.txt   (first time only)
echo                python LocalDashboard\dashboard.py
echo                then open http://localhost:8050
echo.
echo Follow logs:   docker compose logs -f tracking-papagou tracking-roupaki
echo                docker compose logs -f spark-analytics
echo                docker compose logs -f spark-alerts
echo.
echo Stop:          docker compose down
