# diva_brain.py - LLM Brain with LLM-native Tool Calling
from ollama._client import Client, Message
import asyncio
import json
from typing import Dict, Any
from diva_mcp import diva_mcp, SYSTEM_PROMPT

model_name = "llama3.2"


class DivaBrain:
    """
    Local LLM brain using Llama3.2 with LLM-native MCP tool integration.
    LLM decides when to use tools via function calling.
    """
    
    def __init__(self, model_name: str = model_name, debug: bool = False, use_mcp: bool = True):
        self.model_name = model_name
        self.debug = debug
        self._client = Client()
        self._stop_flag = False
        self._use_mcp = use_mcp
        self._mcp = diva_mcp
        
        self.system_prompt = SYSTEM_PROMPT
    
    def _parse_tool_calls(self, response) -> list:
        """Parse tool calls from Ollama response format"""
        if hasattr(response, 'message') and hasattr(response.message, 'tool_calls'):
            tool_calls = response.message.tool_calls
            if tool_calls:
                parsed = []
                for tc in tool_calls:
                    if hasattr(tc, 'function'):
                        func = tc.function
                        name = func.name if hasattr(func, 'name') else str(func)
                        
                        args = {}
                        if hasattr(func, 'arguments'):
                            args_raw = func.arguments
                            if isinstance(args_raw, str):
                                try:
                                    args = json.loads(args_raw)
                                except:
                                    args = {"query": args_raw}
                            elif isinstance(args_raw, dict):
                                # Handle nested format: {'param': {'type': ..., 'value': ...}}
                                for k, v in args_raw.items():
                                    if isinstance(v, dict) and 'value' in v:
                                        args[k] = v['value']
                                    else:
                                        args[k] = v
                        
                        parsed.append({"name": name, "arguments": args})
                return parsed
        return []
    
    def stream(self, text_input: str):
        """Stream response with LLM-native tool calling"""
        self._stop_flag = False
        
        if self._use_mcp:
            tools = self._mcp.get_tools_for_llm()
            
            response = self._client.chat(
                model=self.model_name,
                messages=[
                    Message(role="system", content=self.system_prompt),
                    Message(role="user", content=text_input)
                ],
                tools=tools,
                stream=False
            )
            
            tool_calls = self._parse_tool_calls(response)
            
            if tool_calls:
                # Show thinking
                tool_names = [tc["name"] for tc in tool_calls]
                yield f"Let me check that for you...\n\n"
                
                # Execute tools silently
                results = {}
                for tc in tool_calls:
                    name = tc["name"]
                    args = tc["arguments"]
                    result = asyncio.run(self._mcp.call_tool(name, args))
                    results[name] = result
                
                # Build clean results message
                tool_results_text = "\n".join([
                    f"{name}: {result}"
                    for name, result in results.items()
                ])
                
                # LLM generates response with tool results
                response = self._client.chat(
                    model=self.model_name,
                    messages=[
                        Message(role="system", content=self.system_prompt),
                        Message(role="user", content=text_input),
                        Message(role="tool", content=tool_results_text)
                    ],
                    stream=True
                )
                
                for token in response:
                    if self._stop_flag:
                        break
                    content = token.get('message', {}).get('content', '')
                    if content:
                        yield content
                return
        
        # No tools needed - normal response
        yield ""
        
        response = self._client.chat(
            model=self.model_name,
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=text_input)
            ],
            stream=True
        )
        
        for token in response:
            if self._stop_flag:
                break
            content = token.get('message', {}).get('content', '')
            if content:
                yield content
    
    def stop(self):
        """Interrupt streaming"""
        self._stop_flag = True


if __name__ == "__main__":
    brain = DivaBrain(debug=False, use_mcp=True)
    
    print("\n=== Test: Weather ===")
    for token in brain.stream("What's the weather in Kathmandu ?"):
        print(token, end="", flush=True)
