@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto :failed
set "PYTHON=%CD%\.venv\Scripts\python.exe"

rem Configuration comes from .env (or explicitly supplied environment variables).
rem Do not silently override the Qwen output/token budgets.
set "PORT=8501"
echo.
echo ============================================================
echo   DAP Life Events Assistant - Windows Run

echo ============================================================
echo.
echo [STEP 1/5] Checking virtual environment and Python...
if not exist "%PYTHON%" (
    echo [ERROR] Missing virtual environment. Run SETUP.bat first.
    goto :failed
)
"%PYTHON%" -c "import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15); print('[OK] Python', sys.version.split()[0])"
if errorlevel 1 goto :failed

echo.
echo [STEP 2/5] Checking Streamlit and LangGraph dependencies...
"%PYTHON%" -c "import streamlit, langgraph; print('[OK] Application dependencies are importable.')"
if errorlevel 1 (
    echo [ERROR] Required packages are missing. Run SETUP.bat again.
    goto :failed
)

echo.
echo [STEP 3/5] Checking LLM configuration and Ollama...
"%PYTHON%" -c "from dap_assistant.settings import Settings; Settings(); print('[OK] .env configuration validated.')"
if errorlevel 1 goto :failed
"%PYTHON%" scripts\launcher_support.py runtime-status
"%PYTHON%" scripts\launcher_support.py dummy-check >nul 2>&1
if not errorlevel 1 (
    echo [OK] Dummy mode: Ollama is not required.
    goto :documents
)
"%PYTHON%" scripts\launcher_support.py ollama-check
if not errorlevel 1 goto :documents
if errorlevel 3 (
    echo [INFO] Ollama is not reachable. Trying the existing local installation...
    "%PYTHON%" scripts\launcher_support.py ollama-start
    if errorlevel 1 goto :offer_dummy
    "%PYTHON%" scripts\launcher_support.py ollama-check
    if not errorlevel 1 goto :documents
)
goto :offer_dummy

:offer_dummy
echo [WARN] Ollama or the configured model is unavailable.
echo [INFO] For Ollama mode run: ollama serve; then check the model name in .env.
echo [INFO] Check the exact configured model with: "%PYTHON%" scripts\launcher_support.py ollama-check
choice /C YN /N /M "Use dummy mode for this run only? [Y/N]: "
if errorlevel 2 goto :failed
set "LLM_PROVIDER=dummy"
echo [OK] Using deterministic dummy model for this run. No .env changes made.

:documents
echo.
echo [STEP 4/5] Checking indexed documents and web port...
"%PYTHON%" scripts\launcher_support.py index-check
if errorlevel 1 (
    echo [WARN] Documents or vector index are incomplete.
    echo [INFO] Run: .venv\Scripts\python.exe scripts\download_documents.py --index
    echo [INFO] The UI will still open and show the missing-data status.
)
"%PYTHON%" scripts\launcher_support.py port-check %PORT%
if errorlevel 1 goto :failed
if not exist "logs" mkdir "logs"

echo.
echo [STEP 5/5] Starting Streamlit in THIS terminal...
echo [INFO] The browser opens automatically when http://localhost:%PORT% is ready.
echo [INFO] Press Ctrl+C to stop the application. Streamlit errors appear below.
"%PYTHON%" scripts\start_streamlit.py
if errorlevel 1 goto :failed
echo [OK] Streamlit stopped normally.
exit /b 0

:failed
echo.
echo [ERROR] RUN.bat could not start the application. Review the message above.
echo [INFO] If the browser did not open, also inspect logs\browser-launch.log
if not defined CI pause
exit /b 1
