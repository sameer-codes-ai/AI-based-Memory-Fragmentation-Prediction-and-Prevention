@echo off
title PREFRAG - Live Interactive Demo
echo ===================================================================
echo   Starting PREFRAG Live Memory Fragmentation & Defragmentation Demo
echo ===================================================================
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe scripts\live_demo.py
) else (
    python scripts\live_demo.py
)
pause
