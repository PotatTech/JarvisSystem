import urllib.parse
import numpy as np
import tempfile
import wave
import os
import re
import json
import time
import datetime
import traceback
import subprocess

# ---------- OpenRouter через httpx ----------
try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False
    print("⚠️ httpx не установлен. Установи: pip install httpx")

# ---------- Переводчик ----------
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_OK = True
except ImportError:
    TRANSLATOR_OK = False
    print("⚠️ deep-translator не установлен. Установи: pip install deep-translator")

# ---------- Аудио ----------
try:
    import sounddevice as sd
    SD_OK = True
except Exception as e:
    print(f"❌ sounddevice: {e}")
    SD_OK = False

try:
    import soundfile as sf
    SF_OK = True
except Exception as e:
    print(f"❌ soundfile: {e}")
    SF_OK = False

# ---------- Sherpa-ONNX JARVIS TTS ----------
try:
    import sherpa_onnx
    SHERPA_OK = True
except ImportError:
    SHERPA_OK = False
    print("⚠️ sherpa-onnx не установлен. Установи: pip install sherpa-onnx")

# ---------- Whisper ----------
WHISPER_BACKEND = None
try:
    from faster_whisper import WhisperModel
    WHISPER_BACKEND = "faster"
except ImportError:
    try:
        import whisper
        WHISPER_BACKEND = "openai"
    except ImportError:
        WHISPER_BACKEND = None

# ================== НАСТРОЙКИ ==================
ASSISTANT_NAME = "JARVIS"
WHISPER_MODEL = "base"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6

# Путь к папке с моделью (не к файлу!)
JARVIS_MODEL_DIR = "./jarvis/en/en_GB/jarvis/high"

# ================== OPENROUTER ==================
OPENROUTER_API_KEY = ""
OPENROUTER_MODEL = "openrouter/free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# ===============================================

# ---------- Фильтр галлюцинаций Whisper ----------
HALLUCINATIONS = {
    "продолжение следует",
    "спасибо за просмотр",
    "редактор субтитров",
    "корректор",
    "субтитры",
    "подписывайтесь",
    "ставьте лайк",
    "dimatorzok",
    "яндекс",
    "плейлист",
    "транскрипция",
    "озвучка",
    "перевод",
    "пока",
    "всё",
    "все",
    "стоп",
    "хватит",
    "до свидания",
    "да",
    "нет",
    "угу",
    "ага",
    "ммм",
}

# ---------- Точные команды выхода ----------
EXIT_COMMANDS = {
    "джарвис стоп",
    "джарвис выключись",
    "джарвис пока",
    "джарвис отключись",
    "джарвис выключение",
    "jarvis stop",
    "jarvis power down",
}


class Jarvis:
    def __init__(self):
        # ---------- JARVIS TTS (исправлено: 3 параметра) ----------
        self.tts = None
        if SHERPA_OK:
            model_path = os.path.join(JARVIS_MODEL_DIR, "jarvis-high.onnx")
            tokens_path = os.path.join(JARVIS_MODEL_DIR, "tokens.txt")
            data_dir = os.path.join(JARVIS_MODEL_DIR, "espeak-ng-data")

            if not os.path.exists(model_path):
                print(f"❌ Модель не найдена: {model_path}")
            elif not os.path.exists(tokens_path):
                print(f"❌ tokens.txt не найден: {tokens_path}")
                print("   Запусти: python convert_jarvis.py")
            elif not os.path.exists(data_dir):
                print(f"❌ espeak-ng-data не найден: {data_dir}")
            else:
                try:
                    print("⏳ Загружаю JARVIS голос...")
                    config = sherpa_onnx.OfflineTtsConfig(
                        model=sherpa_onnx.OfflineTtsModelConfig(
                            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                                model=model_path,
                                tokens=tokens_path,
                                data_dir=data_dir,
                            ),
                        ),
                    )
                    self.tts = sherpa_onnx.OfflineTts(config)
                    print("✅ JARVIS голос готов")
                except Exception as e:
                    print(f"❌ JARVIS TTS: {e}")
                    traceback.print_exc()
        else:
            print("⚠️ sherpa-onnx не установлен")

        # ---------- Whisper ----------
        self.whisper_model = None
        if WHISPER_BACKEND == "faster":
            try:
                print(f"⏳ Загружаю faster-whisper {WHISPER_MODEL}...")
                self.whisper_model = WhisperModel(
                    WHISPER_MODEL, device="cpu", compute_type="int8"
                )
                print("✅ Whisper готов")
            except Exception as e:
                print(f"❌ faster-whisper: {e}")
        elif WHISPER_BACKEND == "openai":
            try:
                print(f"⏳ Загружаю openai-whisper {WHISPER_MODEL}...")
                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                print("✅ Whisper готов")
            except Exception as e:
                print(f"❌ openai-whisper: {e}")

        # ---------- HTTP-клиент для OpenRouter ----------
        self.http = None
        if HTTPX_OK and OPENROUTER_API_KEY and "ВСТАВЬ" not in OPENROUTER_API_KEY:
            try:
                self.http = httpx.Client(
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://localhost",
                        "X-Title": "JARVIS Assistant",
                    },
                    timeout=60.0,
                )
                print(f"✅ OpenRouter готов (модель: {OPENROUTER_MODEL})")
            except Exception as e:
                print(f"❌ OpenRouter: {e}")
        else:
            print("⚠️ OpenRouter не настроен — вставь API-ключ")

        self.is_running = True

    # ============================================================
    #   ПЕРЕВОД RU → EN
    # ============================================================
    def translate_to_en(self, text):
        if not TRANSLATOR_OK or not text:
            return None
        if not re.search(r'[а-яА-ЯёЁ]', text):
            return text
        try:
            translated = GoogleTranslator(source='ru', target='en').translate(text)
            if translated:
                return translated
        except Exception as e:
            print(f"⚠️ Перевод: {e}")
        return None

    # ============================================================
    #   НОРМАЛИЗАЦИЯ МАТЕМАТИКИ ДЛЯ TTS
    # ============================================================
    def normalize_for_tts(self, text):
        """Преобразует математические символы в слова, чтобы TTS их прочитал."""
        if not text:
            return text
        text = re.sub(r"(\d+)\s*\^\s*(\d+)", r"\1 to the power of \2", text)
        text = text.replace(" = ", " equals ")
        text = text.replace("=", " equals ")
        text = text.replace(" + ", " plus ")
        text = text.replace("+", " plus ")
        text = text.replace(" - ", " minus ")
        text = text.replace(" * ", " times ")
        text = text.replace(" / ", " divided by ")
        text = text.replace(" × ", " times ")
        text = text.replace(" ÷ ", " divided by ")
        text = text.replace(" % ", " percent ")
        text = text.replace("(", " ").replace(")", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ============================================================
    #   ГОЛОС JARVIS
    # ============================================================
    def speak(self, text):
        print(f"🤖 {ASSISTANT_NAME}: {text}")
        if not self.tts:
            print("⚠️ TTS недоступен — модель не загружена")
            return
        if not SF_OK:
            print("⚠️ soundfile не установлен")
            return
        if re.search(r'[а-яА-ЯёЁ]', text):
            print("⚠️ JARVIS читает только английский — текст содержит кириллицу")
            return

        if SD_OK:
            try:
                sd.stop()
                time.sleep(0.3)
            except Exception:
                pass

        raw_path = None
        out_path = None
        try:
            audio = self.tts.generate(text, sid=0, speed=1.05)
            tmp_raw = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            raw_path = tmp_raw.name
            tmp_raw.close()
            sherpa_onnx.write_wave(raw_path, audio.samples, audio.sample_rate)

            tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out_path = tmp_out.name
            tmp_out.close()

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", raw_path,
                "-af",
                "asetrate=22050*1.05,aresample=22050,"
                "flanger=delay=0:depth=2:regen=50:width=71:speed=0.5,"
                "aecho=0.8:0.88:15:0.5,"
                "highpass=f=200,"
                "treble=g=6",
                out_path
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"⚠️ ffmpeg: {result.stderr[:200]}")
                out_path = raw_path

            data, sr = sf.read(out_path)
            sd.play(data, sr)
            sd.wait()
            sd.stop()
            print("🔊 Озвучено (JARVIS style)")
        except Exception as e:
            print(f"⚠️ speak ошибка: {e}")
            traceback.print_exc()
        finally:
            for p in (raw_path, out_path):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except:
                        pass

    # ---------- Запись ----------
    def record_audio(self, duration=RECORD_SECONDS):
        if not SD_OK:
            return None
        print("🎤 Слушаю...")
        try:
            recording = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            sd.stop()
            time.sleep(0.2)
        except Exception as e:
            print(f"❌ Запись: {e}")
            return None

        if np.abs(recording).max() < 0.005:
            return None

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        audio_int16 = (recording.flatten() * 32767).astype(np.int16)
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())

        return tmp_path

    # ---------- Распознавание ----------
    def transcribe(self, audio_path):
        if not audio_path or not self.whisper_model:
            return ""
        try:
            if WHISPER_BACKEND == "faster":
                segments, _ = self.whisper_model.transcribe(
                    audio_path,
                    language="ru",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                )
                text = " ".join(s.text for s in segments).strip()
            else:
                result = self.whisper_model.transcribe(
                    audio_path, language="ru", fp16=False
                )
                text = result["text"].strip()

            text = text.lower().strip(" .,!?-—")

            if text in HALLUCINATIONS or len(text) < 4:
                print(f"🗑️ Отфильтровано: '{text}'")
                return ""

            return text
        except Exception as e:
            print(f"⚠️ transcribe: {e}")
            return ""
        finally:
            try:
                os.unlink(audio_path)
            except:
                pass

    # ============================================================
    #   ЗАПРОС К OPENROUTER LLM
    # ============================================================
    def ask_llm(self, query):
        if not self.http:
            return None
        try:
            print(f"🌐 [OpenRouter] Запрос к {OPENROUTER_MODEL}...")
            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Ты — JARVIS, голосовой ассистент. "
                            "Отвечай кратко, по делу, на русском языке. "
                            "Максимум 2-3 предложения."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "max_tokens": 300,
            }
            resp = self.http.post(OPENROUTER_URL, json=payload)

            if resp.status_code == 429:
                print("⚠️ OpenRouter: лимит запросов")
                return None
            if resp.status_code == 401:
                print("⚠️ OpenRouter: неверный API-ключ")
                return None
            if resp.status_code == 404:
                print(f"⚠️ OpenRouter: модель '{OPENROUTER_MODEL}' не найдена")
                return None
            if resp.status_code != 200:
                print(f"⚠️ OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            if answer:
                print("✅ OpenRouter ответил")
                answer = re.sub(r"\s+", " ", answer).strip()
                if len(answer) > 400:
                    answer = answer[:400] + "..."
                return answer
        except Exception as e:
            print(f"⚠️ OpenRouter ошибка: {e}")
            traceback.print_exc()
        return None

    # ---------- Логика ----------
    def process(self, cmd):
        if not cmd:
            return

        cmd_clean = cmd.strip()

        # Приветствие
        if any(w in cmd_clean for w in ["привет", "здравствуй", "hello"]):
            self.speak("Hello sir. JARVIS is online.")
            return

        # Время
        if "время" in cmd_clean or "час" in cmd_clean:
            now = datetime.datetime.now().strftime("%H:%M")
            self.speak(f"It is {now}, sir.")
            return

        # Выход — только точная команда
        if cmd_clean in EXIT_COMMANDS:
            self.speak("Powering down, sir.")
            self.is_running = False
            return

        # Чистим запрос от обращения
        query = re.sub(r"^(джарвис|джарвес|jarvis)[\s,\.]*", "", cmd_clean).strip()
        if not query:
            query = cmd_clean

        if query in EXIT_COMMANDS:
            self.speak("Powering down, sir.")
            self.is_running = False
            return

        print("🧠 Думаю...")
        answer_ru = self.ask_llm(query)

        if not answer_ru:
            self.speak("I could not process your request, sir.")
            return

        print(f"📄 Ответ (рус): {answer_ru}")

        answer_en = self.translate_to_en(answer_ru)
        if answer_en:
            answer_en = self.normalize_for_tts(answer_en)
            print(f"🌐 Перевод (en): {answer_en}")
            self.speak(answer_en)
        else:
            self.speak("I have the answer, sir. Check the console.")

    # ---------- Запуск ----------
    def run(self):
        self.speak("JARVIS online. How may I assist you, sir?")
        while self.is_running:
            try:
                audio = self.record_audio()
                if not audio:
                    continue
                cmd = self.transcribe(audio)
                if cmd:
                    print(f"👤 Вы: {cmd}")
                    self.process(cmd)
            except KeyboardInterrupt:
                self.speak("Powering down, sir.")
                break
            except Exception:
                traceback.print_exc()

        if self.http:
            self.http.close()


if __name__ == "__main__":
    Jarvis().run()