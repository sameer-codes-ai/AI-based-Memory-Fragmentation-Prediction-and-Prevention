@echo off
echo =======================================================
echo PREFRAG: Running Test Suite (pytest)
echo =======================================================
.venv\Scripts\python.exe -m pytest tests/test_all.py -v
if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] All tests passed!
) else (
    echo.
    echo [FAILURE] Test suite encountered errors.
)
pause
