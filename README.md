# Diva - Local Voice Assistant with MCP

A privacy-focused voice assistant powered by local LLMs with MCP (Model Context Protocol) tool integration.

## Features

### Voice Recognition
- **STT**: Faster-Whisper for accurate speech-to-text
- **VAD**: Silero VAD for voice activity detection
- **Wake Word**: Listen mode with interrupt support

### LLM Brain
- **Local Models**: Runs Ollama models (gemma3, llama3.2)
- **Streaming**: Token-by-token streaming responses
- **Chain-of-Thought**: Shows reasoning process before answering
- **Conversation History**: Maintains context across interactions

### MCP Tool Integration
- **LLM-Native Tool Calling**: Models decide when to use tools
- **Real-time Data**: Weather, web search from live sources

### Tools Available

#### Weather (`get_weather`)
- Get current weather for any city
- Temperature, conditions, humidity, wind speed
- Uses OpenWeatherMap API

#### Web Search (`web_search`)
- Search the web for real-time information
- Uses Tavily API (1000 free searches/month)

### Thinking Process
The assistant shows step-by-step reasoning:
```
Let me think about this...
The user is asking about...
I should use [tool_name] to get accurate data...
[answer]
```


## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Ollama
```bash
# Download from https://ollama.ai 
ollama pull llama3.2 # or any models
```

### 3. Configure API Keys
Create `.env` file:
```env
OPENWEATHERMAP_API_KEY=your_key
TAVILY_API_KEY=your_key
```
## API Keys Required

Get free API keys:
- OpenWeatherMap: [get-key-from-here](https://openweathermap.org/api)
- Tavily: [get-key-from-here](https://app.tavily.com)

### 4. Run
```bash
python main.py
```

## Usage

### Voice Commands
- Say your question naturally
- Use "bye" or "exit" to stop

### Examples
```
"What's the weather in London?"
"Search for latest tech news"
"Who is the president of USA?"
"Hello, how are you?"
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                  main.py                    │
│  (VAD → STT → Brain → LLM → Response)       │  
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│              diva_brain.py                  │
│  • Detects tool needs                       │
│  • Shows thinking process                   │
│  • Streams tokens                           │
└─────────────────────────────────────────────┘
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│    diva_mcp.py  │  │    Ollama LLM   │
│  • Tool schemas │  │  (llama3.2)     │
│  • Executes     │  │                 │
└─────────────────┘  └─────────────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│              MCP Servers                    │
│  • weather_mcp_server.py                    │
│  • web_search_mcp_server.py                 │
└─────────────────────────────────────────────┘
```



## working on......

- [ ] Add TTS streaming integration (XTTS v2)
- [ ] Add email MCP server
- [ ] Add news MCP server
- [ ] Voice cloning
- [ ] Conversation memory
- [ ] Tool for calendar/tasks

