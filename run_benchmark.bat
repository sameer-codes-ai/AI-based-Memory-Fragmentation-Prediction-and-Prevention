@echo off
echo =======================================================
echo PREFRAG: End-to-End Research Benchmark Pipeline
echo =======================================================
set EVENTS=%1
if "%EVENTS%"=="" set EVENTS=2500
echo Running benchmark with %EVENTS% events per workload...
.venv\Scripts\python.exe scripts/run_all_experiments.py --events %EVENTS%
if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Benchmark finished! Results written to:
    echo   - results\benchmark_report.md
    echo   - results\experiment_summary.json
) else (
    echo.
    echo [FAILURE] Benchmark encountered an error.
)
pause
