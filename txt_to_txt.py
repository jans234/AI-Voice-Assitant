from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from weather import get_weather
import os
from dotenv import load_dotenv

load_dotenv()

PROMPT = """
You are a helpful voice assistant. Keep your responses concise and conversational 
since they will be spoken aloud. Aim for 2-3 sentences unless more detail is requested.
"""

@tool
def get_latest_weather(location_name):
    """
    Returns the latest weather information for a given location.
    """
    return get_weather(location_name)

tools = [get_latest_weather]


def txt_to_txt(txt: str) -> str:
    """
    Process text input through LLM and return response
    OPTIMIZED: Uses faster model by default
    
    Args:
        txt: Input text from user
        
    Returns:
        str: AI generated response
    """
    try:
        # Use faster model: gpt-4o-mini or gpt-3.5-turbo
        model = os.getenv("TEXT_TO_TEXT_MODEL", "gpt-4o-mini")
        
        llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=150,  # Limit response length for voice
            timeout=10,  # Add timeout
        ).bind_tools(tools)
        
        messages = [
            SystemMessage(content=PROMPT),
            HumanMessage(content=txt)
        ]
        
        response = llm.invoke(messages)
        messages.append(response)

        # Handle tool calls
        if response.tool_calls:
            print(f"Tool calls: {response.tool_calls}")
            for tool_call in response.tool_calls:
                if tool_call["name"] == "get_latest_weather":
                    weather_info = get_latest_weather.invoke(tool_call["args"]["location_name"])
                    messages.append(ToolMessage(content=weather_info, tool_call_id=tool_call["id"]))

            response = llm.invoke(messages)

        return response.content
        
    except Exception as e:
        print(f"Error in txt_to_txt: {e}")
        return "I apologize, but I encountered an error processing your request."


async def txt_to_txt_stream(txt: str):
    """
    Stream text response from LLM for faster perceived response
    
    Args:
        txt: Input text from user
        
    Yields:
        str: Chunks of AI response
    """
    try:
        model = os.getenv("TEXT_TO_TEXT_MODEL", "gpt-4o-mini")
        
        llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            max_tokens=150,
            timeout=10,
            streaming=True,  # Enable streaming
        ).bind_tools(tools)
        
        messages = [
            SystemMessage(content=PROMPT),
            HumanMessage(content=txt)
        ]
        
        # Stream the response
        full_response = ""
        has_tool_calls = False
        
        async for chunk in llm.astream(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content
            
            # Check for tool calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                has_tool_calls = True
                break
        
        # If tool calls were made, handle them
        if has_tool_calls:
            messages.append(chunk)
            
            for tool_call in chunk.tool_calls:
                if tool_call["name"] == "get_latest_weather":
                    weather_info = get_latest_weather.invoke(tool_call["args"]["location_name"])
                    messages.append(ToolMessage(content=weather_info, tool_call_id=tool_call["id"]))
            
            # Get final response after tool execution
            async for chunk in llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        
    except Exception as e:
        print(f"Error in txt_to_txt_stream: {e}")
        yield "I apologize, but I encountered an error processing your request."