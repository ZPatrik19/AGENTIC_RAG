@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto :failed
set "PYTHON=%CD%\.venv\Scripts\python.exe"

rem Configuration comes from .env (or explicitly supplied environment variables).
rem Do not silently override the Qwen output/token budgets.

echo.
echo ============================================================
echo   DAP Life Events Assistant - Windows Setup

echo ============================================================
echo.
echo [STEP 1/7] Checking Python and virtual environment...
if exist "%PYTHON%" goto :existing_venv
:choose_python
where py >nul 2>&1
if errorlevel 1 goto :python_without_launcher
for %%V in (3.12 3.13 3.14) do (
    if not exist "%PYTHON%" (
        py -%%V -c "import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15)" >nul 2>&1
        if not errorlevel 1 (
            echo [INFO] Creating .venv using Python %%V...
            py -%%V -m venv ".venv"
        )
    )
)
if exist "%PYTHON%" goto :existing_venv
:python_without_launcher
where python >nul 2>&1
if errorlevel 1 goto :missing_python
python -c "import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15)" >nul 2>&1
if errorlevel 1 goto :missing_python
echo [INFO] Creating .venv using the existing Python installation...
python -m venv ".venv"
if errorlevel 1 goto :failed
:existing_venv
if not exist "%PYTHON%" goto :missing_python
"%PYTHON%" -c "import sys; assert (3, 12) <= sys.version_info[:2] < (3, 15); print('[OK] Python', sys.version.split()[0], 'is ready.')"
if errorlevel 1 goto :old_venv

echo.
echo [STEP 2/7] Checking and installing Python dependencies...
"%PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] Restoring pip in the existing virtual environment...
    "%PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 goto :failed
)
"%PYTHON%" -m pip install -e ".[rag,dev]"
if errorlevel 1 goto :failed
echo [OK] Dependencies are ready. Satisfied packages are not reinstalled.

echo.
echo [STEP 3/7] Preparing local configuration and data directories...
"%PYTHON%" scripts\config\sync_env.py
if errorlevel 1 goto :failed
"%PYTHON%" -c "from dap_assistant.settings import Settings; Settings(); print('[OK] .env configuration validated.')"
if errorlevel 1 goto :failed
for %%D in ("data\raw" "data\interim" "data\processed" "data\vectorstore" "logs") do (
    if not exist "%%~D" mkdir "%%~D"
)
"%PYTHON%" scripts\launcher_support.py runtime-status

echo.
echo [STEP 4/7] Checking existing Ollama installation and local model...
"%PYTHON%" scripts\launcher_support.py dummy-check >nul 2>&1
if not errorlevel 1 (
    echo [OK] Dummy mode is configured. Ollama installation and model downloads are skipped.
    goto :documents
)
"%PYTHON%" scripts\launcher_support.py ollama-check
if not errorlevel 1 goto :ollama_ready
if errorlevel 4 goto :ollama_missing
if errorlevel 3 goto :ollama_offline
if errorlevel 2 goto :ollama_model_missing
goto :failed

:ollama_missing
echo [WARN] Ollama executable was not found. An existing installation will never be reinstalled.
where winget >nul 2>&1
if errorlevel 1 (
    echo [WARN] winget is unavailable. Install Ollama from https://ollama.com/download/windows
    goto :offer_dummy
)
choice /C YN /N /M "Install Ollama via winget now? [Y/N]: "
if errorlevel 2 goto :offer_dummy
winget install --id Ollama.Ollama --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [WARN] Ollama installation failed. Check winget output.
    goto :offer_dummy
)
"%PYTHON%" scripts\launcher_support.py ollama-start
if errorlevel 1 goto :offer_dummy
goto :ollama_model_missing

:ollama_offline
"%PYTHON%" scripts\launcher_support.py ollama-start
if errorlevel 1 goto :offer_dummy
"%PYTHON%" scripts\launcher_support.py ollama-check
if not errorlevel 1 goto :ollama_ready
if errorlevel 2 goto :ollama_model_missing
goto :offer_dummy

:ollama_model_missing
choice /C YN /N /M "Download the configured missing Ollama model now? [Y/N]: "
if errorlevel 2 goto :offer_dummy
"%PYTHON%" scripts\launcher_support.py ollama-pull
if errorlevel 1 goto :offer_dummy
goto :ollama_ready

:ollama_ready
echo [OK] Existing Ollama and model are ready. No reinstall performed.
goto :documents

:offer_dummy
echo [WARN] The Ollama mode is not ready.
choice /C YN /N /M "Continue in dummy mode for this setup/run only? [Y/N]: "
if errorlevel 2 goto :failed
set "LLM_PROVIDER=dummy"
echo [OK] Temporary dummy mode enabled for this process. .env was NOT changed.

echo.
:documents
echo [STEP 5/7] Checking official documents and search index...
"%PYTHON%" scripts\launcher_support.py index-check
if not errorlevel 1 (
    echo [OK] Existing documents and index reused. No download is necessary.
    goto :verify
)
echo [INFO] Running Download - Parse - Clean - Chunk - Embed - Index...
echo [INFO] Official sources and embedding weights require internet on first setup.
"%PYTHON%" scripts\download_documents.py --index --summary
if errorlevel 1 (
    echo [WARN] Download or indexing failed. Existing valid data was preserved.
    echo [WARN] Retry with: .venv\Scripts\python.exe scripts\download_documents.py --index --summary
    choice /C YN /N /M "Open the UI anyway with incomplete documents? [Y/N]: "
    if errorlevel 2 goto :failed
) else (
    echo [OK] Download and indexing stage completed.
)

:verify
echo.
echo [STEP 6/7] Checking Streamlit and LangGraph imports...
"%PYTHON%" -c "import streamlit, langgraph; print('[OK] Streamlit and LangGraph are importable.')"
if errorlevel 1 goto :failed

echo.
echo [STEP 7/7] Launching Streamlit via RUN.bat...
echo [INFO] RUN.bat keeps this terminal open and displays Streamlit errors.
call "%~dp0RUN.bat"
set "RUN_EXIT=%ERRORLEVEL%"
if not "%RUN_EXIT%"=="0" goto :failed
echo [OK] Application stopped normally.
exit /b 0

:old_venv
echo [WARN] Existing .venv is older than Python 3.12.
choice /C YN /N /M "Recreate the incompatible .venv now? [Y/N]: "
if errorlevel 2 goto :failed
rmdir /s /q ".venv"
if errorlevel 1 goto :failed
goto :choose_python

:missing_python
echo [ERROR] Python 3.12-3.14 not found. Install Python 3.12+ or check py -0p.
echo [ERROR] Verify your Python launcher with: py -0p
goto :failed

:failed
echo.
echo [ERROR] Setup or startup did not complete. See the error above.
echo [INFO] You can retry SETUP.bat after fixing the issue.
if not defined CI pause
exit /b 1
