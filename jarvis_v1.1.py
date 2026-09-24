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

# ---------- Tavily ----------
try:
    from tavily import TavilyClient
    TAVILY_OK = True
except ImportError:
    TAVILY_OK = False
    print("⚠️ tavily-python не установлен. Установи: pip install tavily-python")

# ---------- Безопасные импорты ----------
try:
    import sounddevice as sd
    SD_OK = True
except Exception as e:
    print(f"❌ sounddevice: {e}")
    SD_OK = False

try:
    import pyttsx3
    TTS_OK = True
except Exception as e:
    print(f"❌ pyttsx3: {e}")
    TTS_OK = False

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
ASSISTANT_NAME = "Джарвис"
WHISPER_MODEL = "base"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
VOICE_RATE = 170

# ================== TAVILY ==================
TAVILY_API_KEY = "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"
# ===============================================


class Jarvis:
    def __init__(self):
        # ---------- TTS: ищем русский голос ОДИН РАЗ при старте ----------
        self.ru_voice_id = None
        self.voice_name = "unknown"
        if TTS_OK:
            try:
                temp_engine = pyttsx3.init()
                voices = temp_engine.getProperty('voices')
                print("🔊 Доступные голоса:")
                for v in voices:
                    print(f"   - {v.name} | {v.id}")

                for v in voices:
                    vid = v.id.lower()
                    vname = v.name.lower()
                    if ('ru' in vid or 'russian' in vname or
                            'irina' in vname or 'pavel' in vname or
                            'ru-ru' in vid):
                        self.ru_voice_id = v.id
                        self.voice_name = v.name
                        break

                if self.ru_voice_id:
                    print(f"✅ Выбран русский голос: {self.voice_name}")
                else:
                    print("⚠️ Русский голос не найден — будет английский")

                temp_engine.stop()
                del temp_engine
            except Exception as e:
                print(f"⚠️ TTS init: {e}")

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

        # ---------- Tavily ----------
        self.tavily = None
        if TAVILY_OK and TAVILY_API_KEY and "ВСТАВЬ" not in TAVILY_API_KEY:
            try:
                self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
                print("✅ Tavily готов")
            except Exception as e:
                print(f"❌ Tavily: {e}")
        else:
            print("⚠️ Tavily не настроен")

        self.is_running = True

    # ============================================================
    #   ГОЛОС: пересоздаём движок каждый раз + освобождаем sd
    # ============================================================
    def speak(self, text):
        print(f"🤖 {ASSISTANT_NAME}: {text}")
        if not TTS_OK or not text:
            print("⚠️ TTS недоступен")
            return

        # Освобождаем аудио-поток sounddevice (главный фикс)
        if SD_OK:
            try:
                sd.stop()
                time.sleep(0.3)
            except Exception:
                pass

        try:
            # Пересоздаём движок каждый раз — обходит баг pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', VOICE_RATE)
            engine.setProperty('volume', 1.0)

            if self.ru_voice_id:
                engine.setProperty('voice', self.ru_voice_id)

            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine

            print("🔊 Озвучено")
        except Exception as e:
            print(f"⚠️ speak ошибка: {e}")
            traceback.print_exc()

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
            sd.stop()          # освобождаем поток после записи
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
                    audio_path,
                    language="ru",
                    fp16=False,
                    condition_on_previous_text=False,
                )
                text = result["text"].strip()
            return text.lower()
        except Exception as e:
            print(f"⚠️ transcribe: {e}")
            return ""
        finally:
            try:
                os.unlink(audio_path)
            except:
                pass

    # ============================================================
    #   ПОИСК ЧЕРЕЗ TAVILY
    # ============================================================
    def search_tavily(self, query):
        if not self.tavily:
            return None

        try:
            print(f"🌐 [Tavily] Запрос: {query}")
            response = self.tavily.search(
                query=query,
                max_results=3,
                include_answer=True,
                search_depth="basic",
                language="ru",              
                filter_by_language=True,    
            )

            answer = response.get("answer")
            if answer and len(answer) > 10:
                print("✅ Tavily answer")
                answer = re.sub(r"\s+", " ", answer).strip()
                if len(answer) > 400:
                    answer = answer[:400] + "..."
                return answer

            print("ℹ️ Tavily answer пусто → беру первый результат")
            results = response.get("results", [])
            if results:
                text = results[0].get("content", "")
                if text and len(text) > 30:
                    return re.sub(r"\s+", " ", text).strip()

        except Exception as e:
            print(f"⚠️ Tavily ошибка: {e}")
            traceback.print_exc()

        return None

    # ---------- Логика ----------
    def process(self, cmd):
        if not cmd:
            return

        if any(w in cmd for w in ["привет", "здравствуй"]):
            self.speak("Здравствуйте, сэр.")
            return

        if "время" in cmd or "час" in cmd:
            self.speak(f"Сейчас {datetime.datetime.now().strftime('%H:%M')}")
            return

        if any(w in cmd for w in ["стоп", "выключись", "пока", "отключись"]):
            self.speak("Отключаюсь.")
            self.is_running = False
            return

        query = re.sub(r"^(джарвис|джарвес|jarvis)[\s,\.]*", "", cmd).strip()
        for t in ["найди", "загугли", "поищи", "погугли",
                  "что такое", "кто такой", "сколько будет"]:
            if query.startswith(t):
                query = query[len(t):].strip()
                break

        if not query:
            return

        print("🧠 Ищу ответ...")
        answer = self.search_tavily(query)

        if answer:
            self.speak(answer)
        else:
            self.speak("Не удалось найти ответ, сэр.")

    # ---------- Запуск ----------
    def run(self):
        self.speak("Джарвис версия один запущен.")
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
                self.speak("Завершаю.")
                break
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    Jarvis().run()