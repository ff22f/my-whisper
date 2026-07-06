#!/bin/zsh
# Запуск WhisperFlow. В меню-баре появится значок микрофона.
cd "$(dirname "$0")"

if pgrep -f "whisper_flow.py" > /dev/null; then
  echo "WhisperFlow уже запущен (значок в меню-баре наверху)."
  exit 0
fi

echo "Запускаю WhisperFlow..."
nohup ./.venv/bin/python whisper_flow.py >> whisper_flow.log 2>&1 &

echo ""
echo "Значок ⌛ в меню-баре сменится на 🎤 - можно диктовать:"
echo "  зажми Option, говори, отпусти - текст вставится."

# Закрываем свое окно Терминала через пару секунд
(sleep 2 && osascript -e 'tell application "Terminal" to close (every window whose name contains "start.command")') > /dev/null 2>&1 &
