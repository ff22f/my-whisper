#!/bin/zsh
# Остановка WhisperFlow.
if pkill -f "whisper_flow.py"; then
  echo "WhisperFlow остановлен."
else
  echo "WhisperFlow не был запущен."
fi
