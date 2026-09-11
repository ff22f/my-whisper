#!/usr/bin/env python3
"""WhisperFlow - локальная диктовка в любом окне (клон Wispr Flow).

Зажала правый Shift - говоришь - отпустила - текст вставился.
Короткое нажатие - запись идет до следующего нажатия. Esc - отмена записи.
Распознавание полностью локальное: mlx-whisper на чипе Apple.
"""

import json
import logging
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_PATH = BASE_DIR / "whisper_flow.log"

SAMPLE_RATE = 16000

# Иконки состояний в меню-баре
ICON_LOADING = "⌛"
ICON_IDLE = "🎤"
ICON_RECORDING = "🔴"
ICON_BUSY = "⏳"

# Состояния
IDLE, RECORDING, BUSY = "idle", "recording", "busy"

# Клавиатура может прислать Option любым из этих кодов (у Сони правый
# Option приходит как общий Key.alt <58>), поэтому принимаем всю семью.
ALT_KEYS = {
    getattr(keyboard.Key, name)
    for name in ("alt", "alt_l", "alt_r", "alt_gr")
    if hasattr(keyboard.Key, name)
}
# Клавиатура может прислать Shift любым из этих кодов, поэтому принимаем всю семью.
SHIFT_KEYS = {
    getattr(keyboard.Key, name)
    for name in ("shift", "shift_l", "shift_r")
    if hasattr(keyboard.Key, name)
}
# Если зажат другой модификатор - это сочетание клавиш, а не диктовка
GUARD_MODIFIERS = {
    getattr(keyboard.Key, name)
    for name in (
        "cmd", "cmd_l", "cmd_r",
        "ctrl", "ctrl_l", "ctrl_r",
        "alt", "alt_l", "alt_r", "alt_gr",
    )
    if hasattr(keyboard.Key, name)
}

# Whisper галлюцинирует на тишине - типичный мусор отсекаем
JUNK_PHRASES = {
    "субтитры сделал dimatorzok",
    "субтитры создавал dimatorzok",
    "продолжение следует...",
    "спасибо за просмотр!",
    "спасибо за просмотр",
    "редактор субтитров а.семкин корректор а.егорова",
    "thank you.",
    "thanks for watching!",
    "you",
}

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s",
)
log = logging.getLogger("whisper_flow")


def load_config() -> dict:
    log.info("Загрузка конфигурации из %s", CONFIG_PATH)
    defaults = {
        "hotkey": "shift_r",
        "model": "mlx-community/whisper-large-v3-turbo",
        "language": "auto",
        "sounds": True,
        "append_space": True,
        "hold_threshold_sec": 0.5,
        "max_recording_sec": 300,
    }
    try:
        config_data = json.loads(CONFIG_PATH.read_text("utf-8"))
        defaults.update(config_data)
        log.info("Конфигурация загружена успешно")
    except FileNotFoundError:
        log.warning("Файл конфигурации не найден, используются значения по умолчанию")
    except Exception as e:
        log.error("Ошибка при загрузке конфигурации: %s", e)
    return defaults


def save_config(cfg: dict) -> None:
    log.debug("Сохранение конфигурации в %s", CONFIG_PATH)
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
        log.debug("Конфигурация сохранена успешно")
    except Exception as e:
        log.error("Ошибка при сохранении конфигурации: %s", e)


def play_sound(name: str, enabled: bool) -> None:
    if enabled:
        log.debug("Воспроизведение звука: %s", name)
        try:
            subprocess.Popen(
                ["afplay", f"/System/Library/Sounds/{name}.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.warning("Не удалось воспроизвести звук %s: %s", name, e)


def get_clipboard() -> str:
    try:
        result = subprocess.run(["pbpaste"], capture_output=True, timeout=3)
        log.debug("Буфер обмена прочитан (%d байт)", len(result.stdout))
        return result.stdout.decode("utf-8", "replace")
    except Exception as e:
        log.warning("Ошибка при чтении буфера обмена: %s", e)
        return ""


def set_clipboard(text: str) -> None:
    log.debug("Запись в буфер обмена (%d символов)", len(text))
    try:
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), timeout=3)
        log.debug("Буфер обмена обновлен успешно")
    except Exception as e:
        log.error("Ошибка при записи в буфер обмена: %s", e)


class WhisperFlowApp(rumps.App):
    def __init__(self):
        super().__init__(ICON_LOADING, quit_button=None)
        self.cfg = load_config()

        self.state = IDLE
        self.lock = threading.Lock()
        self.model_ready = False

        self.chunks: list[np.ndarray] = []
        self.frames_recorded = 0
        self.stream: sd.InputStream | None = None
        self.press_time = 0.0
        self.press_started_recording = False
        self.auto_stopped = False

        self.last_text = ""
        self.kb = keyboard.Controller()

        # --- меню ---
        self.item_status = rumps.MenuItem("Home", callback=self.menu_home)
        self.item_copy = rumps.MenuItem(
            "Скопировать последний текст", callback=self.menu_copy_last
        )
        self.lang_items = {
            "auto": rumps.MenuItem("Язык: авто", callback=self.menu_lang),
            "ru": rumps.MenuItem("Язык: русский", callback=self.menu_lang),
            "en": rumps.MenuItem("Язык: английский", callback=self.menu_lang),
        }
        self.item_sounds = rumps.MenuItem("Звуки", callback=self.menu_sounds)
        self.item_sounds.state = 1 if self.cfg["sounds"] else 0
        self.item_polish = rumps.MenuItem(
            "Авторедактура (исправлять текст)", callback=self.menu_polish
        )
        self.item_polish.state = 1 if self.cfg.get("polish") else 0
        self.menu = [
            self.item_status,
            None,
            self.item_copy,
            None,
            *self.lang_items.values(),
            self.item_polish,
            self.item_sounds,
            None,
            rumps.MenuItem("Выход", callback=self.menu_quit),
        ]
        self._sync_lang_menu()

        # Глобальный слушатель клавиатуры (нужен «Мониторинг ввода»)
        hotkey_name = self.cfg["hotkey"]
        log.info("Инициализация приложения. Хоткей: %s, модель: %s", hotkey_name, self.cfg["model"])
        if hotkey_name.startswith("shift"):
            self.hotkeys = SHIFT_KEYS
        elif hotkey_name.startswith("alt"):
            self.hotkeys = ALT_KEYS
        else:
            self.hotkeys = {getattr(keyboard.Key, hotkey_name, keyboard.Key.shift_r)}
        self.held_modifiers: set = set()
        self.hotkey_down = False
        self.listener = keyboard.Listener(
            on_press=self.on_press, on_release=self.on_release
        )
        self.listener.start()

        threading.Thread(target=self.warmup_model, daemon=True).start()

    # ---------- модель ----------

    def warmup_model(self):
        """Грузим веса заранее, чтобы первая диктовка не тормозила."""
        log.info("Начало загрузки модели: %s", self.cfg["model"])
        try:
            import mlx_whisper

            log.info("Вызов mlx_whisper.transcribe для прогрева модели")
            mlx_whisper.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                path_or_hf_repo=self.cfg["model"],
            )
            self.model_ready = True
            self.title = ICON_IDLE
            self.item_status.title = "Home"
            play_sound("Glass", self.cfg["sounds"])
            log.info("Модель загружена успешно")
            if self.cfg.get("polish"):
                log.info("Прогрев polish модели: %s", self.cfg.get("polish_model"))
                self.polish("прогрев")  # заранее грузим редактора в память
        except Exception:
            log.exception("Не удалось загрузить модель")
            self.title = "⚠️"
            self.item_status.title = "Home"

    # ---------- запись ----------

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            log.warning("audio_callback: статус %s", status)
        self.chunks.append(indata.copy())
        self.frames_recorded += frames
        if (
            self.frames_recorded > self.cfg["max_recording_sec"] * SAMPLE_RATE
            and not self.auto_stopped
        ):
            log.info("Достигнут лимит записи (%d сек) - автоостановка", self.cfg["max_recording_sec"])
            self.auto_stopped = True
            threading.Thread(target=self.stop_and_transcribe, daemon=True).start()

    def start_recording(self):
        log.info("Инициализация записи")
        self.chunks = []
        self.frames_recorded = 0
        self.auto_stopped = False
        try:
            log.info("Создание InputStream (sample_rate=%d, channels=1)", SAMPLE_RATE)
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
            self.state = RECORDING
            self.title = ICON_RECORDING
            play_sound("Pop", self.cfg["sounds"])
            log.info("Запись началась успешно")
        except Exception:
            log.exception("Ошибка при старте записи")
            self.state = IDLE
            self.title = ICON_IDLE

    def _close_stream(self) -> np.ndarray:
        if self.stream is not None:
            log.info("_close_stream: остановка и закрытие потока")
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.chunks:
            log.warning("_close_stream: нет записанных фрагментов (chunks)")
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self.chunks)[:, 0]
        log.info("_close_stream: собрано аудио (%d фрагментов, %.2f сек)", len(self.chunks), len(audio) / SAMPLE_RATE)
        return audio

    def cancel_recording(self, silent=False):
        with self.lock:
            if self.state != RECORDING:
                log.info("Отмена записи: состояние не RECORDING (%s)", self.state)
                return
            log.info("Закрытие потока записи")
            self._close_stream()
            self.state = IDLE
        self.title = ICON_IDLE
        if not silent:
            play_sound("Basso", self.cfg["sounds"])
        log.info("Запись отменена%s", " (шорткат с Shift)" if silent else " (Esc)")

    def stop_and_transcribe(self):
        with self.lock:
            if self.state != RECORDING:
                log.info("stop_and_transcribe: состояние не RECORDING (%s)", self.state)
                return
            log.info("Остановка записи и подготовка к транскрипции")
            audio = self._close_stream()
            self.state = BUSY
        self.title = ICON_BUSY
        play_sound("Bottle", self.cfg["sounds"])
        try:
            log.info("Начало транскрипции (длительность аудио: %.2f сек)", len(audio) / SAMPLE_RATE)
            self.transcribe_and_paste(audio)
        except Exception:
            log.exception("Ошибка распознавания")
        finally:
            with self.lock:
                self.state = IDLE
            self.title = ICON_IDLE
            log.info("Транскрипция завершена, возврат в IDLE")

    # ---------- распознавание и вставка ----------

    def transcribe_and_paste(self, audio: np.ndarray):
        duration = len(audio) / SAMPLE_RATE
        log.info("transcribe_and_paste: длительность аудио %.2f сек", duration)
        if duration < 0.3:
            log.info("Слишком короткая запись (%.2f c) - пропускаю", duration)
            return
        if np.abs(audio).max() < 0.01:
            log.info("Тишина - пропускаю")
            return

        import mlx_whisper

        lang = None if self.cfg["language"] == "auto" else self.cfg["language"]
        t0 = time.time()
        log.info("Вызов mlx_whisper.transcribe (язык: %s)", lang if lang else "auto")
        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.cfg["model"],
                language=lang,
                # Пример оформленного текста настраивает модель ставить
                # пунктуацию и заглавные буквы
                initial_prompt=self.cfg.get("initial_prompt") or None,
            )
        except Exception:
            log.exception("Ошибка при вызове mlx_whisper.transcribe")
            return
        
        text = result["text"].strip()
        log.info(
            "Распознано за %.1f c (%.1f c аудио, язык %s): %r",
            time.time() - t0,
            duration,
            result.get("language"),
            text,
        )

        if not text or text.lower().strip(" .!") in {
            p.strip(" .!") for p in JUNK_PHRASES
        }:
            log.info("Пустой или мусорный результат - пропускаю")
            return

        if self.cfg.get("polish"):
            log.info("Вызов polish для текста")
            text = self.polish(text)

        self.last_text = text
        log.info("Вставка текста в буфер обмена")
        self.paste_text(text + (" " if self.cfg["append_space"] else ""))
        play_sound("Glass", self.cfg["sounds"])

    def polish(self, text: str) -> str:
        """Авторедактура через локальную модель в Ollama.

        Любая ошибка (Ollama не запущена, таймаут) - возвращаем текст как есть,
        диктовка важнее редактуры.
        """
        log.info("polish: начало обработки текста (%d символов)", len(text))
        try:
            t0 = time.time()
            req_data = {
                "model": self.cfg["polish_model"],
                "messages": [
                    {"role": "system", "content": self.cfg["polish_prompt"]},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
                "keep_alive": "2h",
            }
            log.info("polish: отправка запроса к %s (модель: %s)", 
                     "http://localhost:11434/api/chat", self.cfg["polish_model"])
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                json.dumps(req_data).encode(),
                headers={"Content-Type": "application/json"},
            )
            reply = json.loads(urllib.request.urlopen(req, timeout=60).read())
            polished = reply["message"]["content"].strip()
            if not polished:
                log.warning("polish: пустой ответ от модели")
                return text
            log.info("Редактура за %.1f c: %r", time.time() - t0, polished)
            return polished
        except Exception as e:
            log.warning("Редактура недоступна (%s) - вставляю как есть", e)
            return text

    def paste_text(self, text: str):
        """Вставка через буфер обмена + Cmd+V, старый буфер возвращаем на место."""
        log.info("paste_text: начало вставки (%d символов)", len(text))
        try:
            old_clipboard = get_clipboard()
            log.info("paste_text: сохранение старого буфера (%d символов)", len(old_clipboard) if old_clipboard else 0)
            set_clipboard(text)
            log.info("paste_text: текст помещен в буфер обмена")
            time.sleep(0.15)  # даем системе отпустить модификаторы хоткея
            # Жмем физическую клавишу V по коду (kVK_ANSI_V = 9), а не символ 'v' -
            # иначе на русской раскладке Cmd+V не срабатывает
            v_key = keyboard.KeyCode.from_vk(9)
            with self.kb.pressed(keyboard.Key.cmd):
                self.kb.press(v_key)
                self.kb.release(v_key)
            log.info("paste_text: отправлена комбинация Cmd+V")
            time.sleep(0.6)  # приложение должно успеть прочитать буфер до отката
            if old_clipboard:
                set_clipboard(old_clipboard)
                log.info("paste_text: восстановлен старый буфер обмена")
        except Exception:
            log.exception("Ошибка при вставке текста")

    # ---------- хоткей ----------

    def on_press(self, key):
        # Отладка: пишем в лог только служебные клавиши (не буквы), чтобы
        # понять, каким кодом приходит хоткей. Включается в config.json.
        if self.cfg.get("debug") and not isinstance(key, keyboard.KeyCode):
            log.info("DEBUG нажата клавиша: %r (state=%s)", key, self.state)
        if key == keyboard.Key.esc:
            if self.state == RECORDING:
                log.info("Нажат Esc во время записи - отмена")
                self.cancel_recording()
            return
        # Хоткей не должен считаться блокирующим модификатором
        if key in GUARD_MODIFIERS and key not in self.hotkeys:
            self.held_modifiers.add(key)
        if key not in self.hotkeys:
            # Другая клавиша, пока Shift еще зажат = это сочетание клавиш,
            # а не диктовка - тихо отменяем случайно начатую запись
            if (
                self.hotkey_down
                and self.state == RECORDING
                and self.press_started_recording
            ):
                log.info("Обнаружено сочетание клавиш во время записи - отмена")
                self.cancel_recording(silent=True)
            return
        self.hotkey_down = True
        with self.lock:
            if self.state == BUSY or not self.model_ready:
                log.info("on_press: модель не готова или состояние BUSY - игнорирование нажатия")
                return
            if self.state == IDLE:
                # Исключаем сам хоткей из проверки модификаторов
                other_modifiers = self.held_modifiers - self.hotkeys
                if other_modifiers:
                    log.info("on_press: зажат другой модификатор - игнорирование")
                    return  # зажат Cmd/Ctrl/Alt - это шорткат, не диктовка
                self.press_time = time.time()
                self.press_started_recording = True
                log.info("on_press: начало записи (хоткей: %s)", key)
                self.start_recording()
                return
            # state == RECORDING: второе нажатие в режиме переключателя
            self.press_started_recording = False
            log.info("on_press: второе нажатие - остановка записи")
        threading.Thread(target=self.stop_and_transcribe, daemon=True).start()

    def on_release(self, key):
        # Хоткей не должен считаться блокирующим модификатором
        if key in GUARD_MODIFIERS and key not in self.hotkeys:
            self.held_modifiers.discard(key)
        if key not in self.hotkeys:
            return
        self.hotkey_down = False
        if self.state != RECORDING or not self.press_started_recording:
            log.info("on_release: состояние не RECORDING или press_started_recording=False")
            return
        held = time.time() - self.press_time
        if held >= self.cfg["hold_threshold_sec"]:
            # режим рации: отпустила - распознаем
            log.info("on_release: удержание %.2f сек >= порога - запуск транскрипции", held)
            threading.Thread(target=self.stop_and_transcribe, daemon=True).start()
        else:
            # короткий тап: переключатель, запись продолжается
            log.info("Режим переключателя: запись до следующего нажатия (удержание %.2f сек)", held)

    # ---------- меню ----------

    def _sync_lang_menu(self):
        for code, item in self.lang_items.items():
            item.state = 1 if self.cfg["language"] == code else 0

    def menu_lang(self, sender):
        for code, item in self.lang_items.items():
            if item is sender:
                self.cfg["language"] = code
                log.info("Язык изменен на: %s", code)
        self._sync_lang_menu()
        save_config(self.cfg)
        log.info("Конфигурация сохранена")

    def menu_polish(self, sender):
        self.cfg["polish"] = not self.cfg.get("polish")
        sender.state = 1 if self.cfg["polish"] else 0
        log.info("Polish переключен: %s", self.cfg["polish"])
        save_config(self.cfg)

    def menu_sounds(self, sender):
        self.cfg["sounds"] = not self.cfg["sounds"]
        sender.state = 1 if self.cfg["sounds"] else 0
        log.info("Звуки переключены: %s", self.cfg["sounds"])
        save_config(self.cfg)

    def menu_copy_last(self, _):
        if self.last_text:
            set_clipboard(self.last_text)
            log.info("Последний текст скопирован в буфер обмена (%d символов)", len(self.last_text))
        else:
            log.info("menu_copy_last: нет текста для копирования")

    def menu_home(self, _):
        """Обработчик кнопки Home - возвращает статус приложения."""
        log.info("Нажата кнопка Home. Статус: %s, модель готова: %s", self.state, self.model_ready)

    def menu_quit(self, _):
        log.info("Завершение работы приложения")
        self.listener.stop()
        rumps.quit_application()


if __name__ == "__main__":
    WhisperFlowApp().run()
