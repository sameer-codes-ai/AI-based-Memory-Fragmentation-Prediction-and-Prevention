@echo off
echo =======================================================
echo PREFRAG: Recording Live Memory Traces (tracemalloc)
echo =======================================================
.venv\Scripts\python.exe scripts/record_live_trace.py
pause
