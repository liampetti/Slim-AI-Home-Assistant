# Slim AI Home Assistant
> **Super tiny AI home voice assistant that can run with/without Qwen3 models for added functionality.**

---

## 🚀 Quick Start

### Launch in Docker
Run the entire pipeline with a single command:

```
docker compose up --build
```


### SearXNG Configuration
To enable JSON responses for the Web Search tool:
1. Start the container once to generate config files.
2. Stop the container.
3. Edit `searxng_data/settings.yml` to include `json` in the formats list:


```
search:
 formats:
    - html
    - json
```

---

## ✨ What's New

*   **New ASR Model:** Tiny but powerful speech recognition with **Moonshine ASR**.
*   **New TTS Model:** High-quality text-to-speech using **Kokoro TTS**.
*   **Simplified Architecture:** Full pipeline runs in a single script (`app.py`) with no bloat.
*   **Optimized Performance:** Qwen3 AI model is now optional for ultra-low resource edge devices.

### 🧹 Removed Features
*   **Knowledge Graph & Voice ID:** Removed to reduce compute overhead and complexity.
*   **Wyoming Protocol:** Switched to a monolithic app approach to remove third-party reliance and improve efficiency.
*   **Complex Audio Processing:** Echo cancellation and noise suppression are now offloaded to the audio device and ASR model.

---

## 💡 Features & Capabilities

### 🏠 Core Automation (No AI Required)
Basic home control functions run instantly without LLM inference.

| Action | Voice Command Examples |
| :--- | :--- |
| **Music Control** | "Play music", "Stop", "Pause", "Skip", "Resume" |
| **Timers** | "Set timer for 10 minutes", "Get timers" |
| **Time Check** | "What time is it?" |

### 🤖 AI Tools
When enabled, the AI agent can access the following integrations:
*   **Climate:** Airtouch HVAC control
*   **Organization:** Google Calendar integration
*   **Lighting:** Philips Hue control
*   **Media:** Spotify music control & Pioneer AVR (eISCP)
*   **Appliances:** LG ThinQ Dishwasher checks & TV (WebOS) control
*   **Information:** Web Search (SearXNG) & Weather (BOM Australia)

---

## 🔗 References & Similar Projects
*   [Rhasspy](https://github.com/rhasspy)
*   [Home Assistant](https://github.com/home-assistant)
*   [OpenVoiceOS](https://github.com/OpenVoiceOS)

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
