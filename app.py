import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re
import queue
import threading
import time
import random
from collections import deque
from typing import Optional, Tuple, List, Dict

import numpy as np
import sounddevice as sd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, MoonshineForConditionalGeneration, pipeline
from kokoro import KPipeline

from utils.system_prompts import getIntentSystemPrompt, getChatSystemPrompt
from utils.intent_catch import catchAll # Regex Intent Catch
import utils.intents as intents


#########################
# Moonshine ASR
#########################
ASR_MODEL_NAME = "UsefulSensors/moonshine-tiny"
SAMPLE_RATE = 16000

CHUNK_DURATION_MS = 200           # callback slice
SILENCE_DURATION_MS = 1000        # how long silence to consider “end of utterance”
MIN_UTTERANCE_MS = 1500           # min speech length to send
MAX_UTTERANCE_MS = 10000          # max speech length to send
SILENCE_THRESHOLD = 0.001         # tweak for your mic/room

asr_device = "cuda" if torch.cuda.is_available() else "cpu"
# asr_device = "cpu"
asr_dtype = torch.float16 if asr_device == "cuda" else torch.float32

def load_moonshine():
    logger.info(f"Loading {ASR_MODEL_NAME} on {asr_device}...")
    processor = AutoProcessor.from_pretrained(ASR_MODEL_NAME)
    asr_model = MoonshineForConditionalGeneration.from_pretrained(ASR_MODEL_NAME).to(
        device=asr_device,
        dtype=asr_dtype,
    ) # type: ignore

    asr_pipe = pipeline(
        task="automatic-speech-recognition", # type: ignore
        model=asr_model,
        tokenizer=processor.tokenizer, # type: ignore
        feature_extractor=processor.feature_extractor, # type: ignore
        device=0 if asr_device == "cuda" else -1, # type: ignore
        dtype=asr_dtype, # type: ignore
    )

    return asr_pipe

frames_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
silence_chunks_needed = max(1, int(SILENCE_DURATION_MS / CHUNK_DURATION_MS))
min_utterance_samples = int(SAMPLE_RATE * MIN_UTTERANCE_MS / 1000)
max_utterance_samples = int(SAMPLE_RATE * MAX_UTTERANCE_MS / 1000)

audio_buffer = deque()  # holds 1D float32 arrays
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
running = True
transcribing = True


#########################
# Qwen3 AI
#########################
# AI_MODEL_NAME = "" # Optional: Empty string if not using AI (regex capture only)
# AI_MODEL_NAME = "Qwen/Qwen3-0.6B" # Best AI option for low resource, CPU only
AI_MODEL_NAME = "Qwen/Qwen3-1.7B-FP8" 
# AI_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507-FP8" # 
THINK_END_TOKEN_ID = 151668  # </think>

ai_device = "cuda" if torch.cuda.is_available() else "cpu"
# ai_device = "cpu"
ai_dtype = torch.float16 if ai_device == "cuda" else torch.float32

def load_qwen3():
    logger.info(f"Loading {AI_MODEL_NAME} on {ai_device}...")
    tokenizer = AutoTokenizer.from_pretrained(AI_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        AI_MODEL_NAME,
        device_map=ai_device,
        dtype=ai_dtype,
    )
    return tokenizer, model


def generate_qwen3(
    tokenizer,
    model,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 32768,
    enable_thinking: bool = True,
) -> Tuple[str, str]:
    """
    Run Qwen3 model with optional system prompt and return (thinking_content, content).
    """
    messages: List[Dict[str, str]] = [] # Reset for each query to keep small and fast, no history
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # Build chat template (Qwen3 supports enable_thinking flag)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,  # default is True for Qwen3
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parse thinking vs final content using </think> token id 151668
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True,
    ).strip("\n")

    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True,
    ).strip("\n")

    return thinking_content, content


############################
# Audio Output
############################
# Create a global pipeline so the model loads only once
kpipeline = KPipeline(repo_id='hexgrad/Kokoro-82M', 
                     lang_code="a", # "a" = auto; use "en-us" etc. if you prefer
                     device="cpu")  

def speak_stream(text: str, voice: str = "af_bella", speed: float = 1.2):
    """
    Generate speech from `text` with Kokoro stream it to speakers.
    """
    generator = kpipeline(
        text, # type: ignore
        voice=voice,
        speed=speed,
        # optional: split on newlines so long text comes in chunks
        split_pattern=r"\n+",
    ) # type: ignore

    sample_rate = 24000  # Kokoro uses 24 kHz

    for _, _, audio in generator:
        # audio: 1D numpy.float32 array
        sd.play(audio, samplerate=sample_rate, blocking=True)


############################
# Transcription
############################
EMOJI_PATTERN = re.compile(
    "["                       # start character class
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F1E0-\U0001F1FF"   # flags
    "\u2600-\u26FF"           # misc symbols
    "\u2700-\u27BF"           # dingbats
    "]+",
    flags=re.UNICODE,
)

def remove_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def is_silent(chunk: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    if chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(chunk ** 2))
    return rms < threshold


def audio_callback(indata, frames, time_info, status):
    if status:
        logger.info(status, flush=True) # type: ignore
    chunk = indata[:, 0].astype(np.float32)

    audio_buffer.append(chunk)


def recorder_thread():
    global audio_buffer
    logger.info("🎤 Starting microphone stream ...")

    silence_counter = 0

    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=frames_per_chunk,
        callback=audio_callback,
    ):
        while running:
            # sleep just enough for one chunk to arrive
            time.sleep(CHUNK_DURATION_MS / 1000.0)

            if transcribing:
                if not audio_buffer:
                    continue

                # Only check silence on the most recent chunk
                last_chunk = audio_buffer[-1]
                if is_silent(last_chunk):
                    silence_counter += 1
                else:
                    silence_counter = 0

                # If we haven't hit silence threshold yet, just keep accumulating until maximum speech length
                if (silence_counter < silence_chunks_needed) and ((len(audio_buffer)*frames_per_chunk) <= max_utterance_samples):
                    continue

                # At this point we have “end of utterance” => concatenate full buffer
                buf = np.concatenate(list(audio_buffer), axis=0)

                # Require minimum length to avoid spamming tiny chunks
                if buf.size >= min_utterance_samples:
                    audio_queue.put(buf.copy())
                    secs = buf.size / SAMPLE_RATE
                    logger.debug(f"\n📝 Enqueued {secs:.2f}s for transcription")

            # Reset state: clear buffer & silence counter
            audio_buffer.clear()
            silence_counter = 0


def fix_json_intent(answer):
    """
    Fixes the AI generated JSON string. 
    This is required for Qwen3-0.6B as it often outputs malformed json.
    """
    try:
        answer = "{"+"".join(answer.split("{")[1:])
        answer = "".join(answer.split("}")[:-1])+"}"
        if len(answer.split(":")) > 3:
            answer = answer.split('"args":')[0]+'"args": ["'+"".join(answer.split(":")[2:]).strip('"} ').replace('"', '')+'"]}'
    except Exception as e:
        logger.error(f"JSON fix failed: {e}")
    return answer


def stream_generator(q):
    """Yields audio from the queue. Stops when it sees 'None'."""
    while True:
        # Blocking get is efficient (sleeps thread until audio arrives)
        audio = q.get()
        
        # Use None as a sentinel value to stop the pipeline cleanly
        if audio is None:
            break
            
        # Yield the numpy array directly to the pipeline
        yield audio


def transcriber_thread():
    global transcribing

    asr_pipe = load_moonshine()

    if len(AI_MODEL_NAME) > 0:
        tokenizer, model = load_qwen3()
        intent_prompt = getIntentSystemPrompt()
        chat_prompt = getChatSystemPrompt()

    logger.info("📜 Transcriber started")
    for result in asr_pipe(
        stream_generator(audio_queue),  # type: ignore
        batch_size=1, # Capture live stream
        generate_kwargs={"max_new_tokens": 256}):
        try:
            text = result.get("text", "").strip()
            if text:
                logger.debug(f"✓ {text}")
                if "alexa" in text.lower():
                    answer = ""
                    # Only gather text after alexa phrase
                    user_prompt = text.lower().split("alexa")[1].strip(",. ").replace('"', '')
                    if len(user_prompt) > 0:
                        transcribing = False
                        run_think = False
                        logger.info(f" - WAKEWORD: {user_prompt}")
                        # Regex intent catch
                        user_prompt = catchAll(user_prompt)
                        if isinstance(user_prompt, dict):
                            # Regex caught the intent, no need for AI
                            logger.info(f"Regex caught intent, loading --> {user_prompt}")
                            answer = intents.handle_intent(user_prompt)
                            logger.info(f" Intent Answer: {answer} ")
                        elif (len(AI_MODEL_NAME) > 0):
                            # AI takes over if enabled
                            logger.info(f"AI intent query, loading --> {user_prompt}")
                            # Check intent with AI model
                            thinking, answer = generate_qwen3(
                                tokenizer,
                                model,
                                user_prompt=user_prompt,
                                system_prompt=intent_prompt,
                                max_new_tokens=512,
                                enable_thinking=False,
                            )

                            logger.info(f" AI Answer: {answer} ")
                            if len(answer.strip('"')) > 0:
                                answer = fix_json_intent(answer)
                                logger.info(f"AI generated intent, loading --> {answer}")
                                answer = intents.handle_intent(answer)
                                logger.info(f" Intent Answer: {answer} ")
                                if "User question:" in answer:
                                    user_prompt = answer
                                    run_think = True
                            else:
                                run_think = True

                            if run_think:
                                speak_stream(random.choice(["Okay, let me think about that.", "Just a second.", "Got it, let me think.", "Let's see."])) 
                                logger.info(f"AI think query, loading --> {user_prompt}")
                                # Think and respond with AI model if needed
                                thinking, answer = generate_qwen3(
                                    tokenizer,
                                    model,
                                    user_prompt=user_prompt,
                                    system_prompt=chat_prompt,
                                    max_new_tokens=512,
                                    enable_thinking=True,
                                )

                                logger.info(f" AI Thinking: {thinking} ")
                                logger.info(f" AI Answer: {answer} ")

                        # Optional: Default answers if nothing else works, otherwise no response
                        if len(answer.strip('"')) == 0:
                            answer = random.choice(["Sorry, can you repeat that", "I don't understand", "Sorry, I didn't hear you properly", "Can you say that again?"])

                        speak_stream(remove_emoji(answer.replace('"', '').replace('*', '')))
                        transcribing = True
                    else: 
                        logger.debug("Nothing transcribed.")
        except Exception as e:
            logger.error(f"Transcription error: {e}")


#############################
# Run with Prompt
#############################
def main():
    global running
    rec_t = threading.Thread(target=recorder_thread, daemon=True)
    tr_t = threading.Thread(target=transcriber_thread, daemon=True)

    rec_t.start()
    tr_t.start()

    logger.info("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Stopping...")
        running = False
        audio_queue.put(None) # Inject a "poison pill" (sentinel) into the queue to stop
        rec_t.join(timeout=2)
        tr_t.join(timeout=2)

if __name__ == "__main__":
    main()
