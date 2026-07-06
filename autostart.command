#!/bin/zsh
# Включить автозапуск диктовки при входе в систему (запустить один раз).
cd "$(dirname "$0")"

osascript -e 'tell application "System Events" to delete (every login item whose name is "start.command")' 2>/dev/null
osascript -e "tell application \"System Events\" to make login item at end with properties {path:\"$PWD/start.command\", hidden:false}"

echo "Готово! WhisperFlow будет запускаться сам при включении Mac."
echo "Отключить: Настройки -> Основные -> Объекты входа -> убрать start.command."
