@echo off
title MOSS-TTS LOCAL UI - Marcelo
color 0b
echo ==========================================
echo    INICIANDO MOSS-TTS ZERO-SHOT
echo ==========================================
echo.
echo [1/2] Ativando ambiente virtual...
call venv\Scripts\activate
echo [2/2] Abrindo interface Gradio...
echo.
python app.py
pause