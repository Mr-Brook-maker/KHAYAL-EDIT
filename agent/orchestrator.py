"""
Agent Orchestrator
The central brain: receives user prompts, maintains conversation memory,
selects tools, executes multi-step editing pipelines, and returns results.
"""

import logging
from pathlib import Path
from typing import Optional

# 1. Agents Imports
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    try:
        from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
    except ImportError:
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.tool_calling.base import create_tool_calling_agent

# 2. Memory Imports (تغطية كافة المسارات الممكنة)
try:
    from langchain.memory import ConversationBufferWindowMemory
except (ModuleNotFoundError, ImportError):
    try:
        from langchain_classic.memory import ConversationBufferWindowMemory
    except (ModuleNotFoundError, ImportError):
        from langchain_community.memory.buffer_window import ConversationBufferWindowMemory

# 3. Core Prompts & Messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from agent.config import settings
from agent.llm_factory import get_llm
from agent.tools import get_all_tools

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI image and video editing assistant with access to a \
powerful suite of professional editing tools.

## YOUR ROLE
- Analyze user requests and decompose them into precise, ordered tool calls
- Chain multiple tools when needed (e.g., remove background → resize → convert)
- Always confirm the output path of each step and use it as input for the next
- Communicate clearly about what you did and the quality of results

## TOOL SELECTION RULES
1. For background removal: always use `remove_background` (local, instant, free)
2. For object removal: prefer `remove_object` for simple cases; use `inpaint_image` + mask for precise control
3. For canvas expansion: use `outpaint_image`
4. For color work: use `adjust_colors` for corrections; `apply_filter` for artistic looks
5. For format changes: use `convert_format` last in any pipeline

## CHAINING RULES
- After each tool call, parse the `output_path` from the JSON response
- Use that path as `image_path` for the next tool in the chain
- If a tool returns `status: error`, diagnose and try an alternative approach

## COMMUNICATION
- Before starting: briefly explain your plan
- After completion: summarize what was done and where the output is saved
- If you need information (like mask path for inpainting): ask the user

## CONSTRAINTS  
- Never fabricate tool outputs — always execute tools and report actual results
- Image size limit: {max_size}MB
- Supported inputs: PNG, JPEG, WEBP, GIF, BMP, TIFF
""".format(max_size=settings.max_image_size_mb)


# ── Memory ────────────────────────────────────────────────────────────────────

def _build_memory() -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=settings.memory_window_size,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )


# ── Prompt Template ───────────────────────────────────────────────────────────

def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ── Agent Builder ─────────────────────────────────────────────────────────────

class ImageEditingAgent:
    """
    High-level agent interface.
    Wraps LangChain AgentExecutor with memory, tools, and structured logging.
    """

    def __init__(self):
        # Ensure output directories exist
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

        self.llm = get_llm()
        self.tools = get_all_tools()
        self.memory = _build_memory() if settings.enable_memory else None
        self.prompt = _build_prompt()

        # Create tool-calling agent (works with Gemini + Groq function calling)
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
        )

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=settings.agent_verbose,
            max_iterations=settings.agent_max_iterations,
            handle_parsing_errors=True,        # Graceful recovery from LLM parse errors
            return_intermediate_steps=True,    # Expose tool calls in response
            early_stopping_method="generate",  # Let LLM decide when done
        )

        logger.info("ImageEditingAgent initialized successfully")

    def run(self, user_prompt: str, image_path: Optional[str] = None) -> dict:
        """
        Execute an editing request.
        
        Args:
            user_prompt: Natural language editing instruction
            image_path: Optional path hint — agent can extract this from prompt too
            
        Returns:
            {
                "output": str,              # Agent's final response
                "steps": list,             # Tool calls made (name, input, output)
                "final_output_path": str,  # Last output image path if applicable
            }
        """
        # Enrich prompt with path context if provided separately
        if image_path and image_path not in user_prompt:
            enriched_prompt = f"Image file: `{image_path}`\n\nTask: {user_prompt}"
        else:
            enriched_prompt = user_prompt

        logger.info(f"Agent processing: {enriched_prompt[:200]}...")

        try:
            result = self.executor.invoke({"input": enriched_prompt})
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return {
                "output": f"Agent encountered an error: {str(e)}",
                "steps": [],
                "final_output_path": None,
                "error": str(e),
            }

        # Extract structured step information
        steps = []
        final_output_path = None

        for action, observation in result.get("intermediate_steps", []):
            import json
            step = {
                "tool": action.tool,
                "input": action.tool_input,
                "output": observation,
            }
            steps.append(step)

            # Track last successful output path for convenience
            try:
                obs_data = json.loads(observation)
                if obs_data.get("status") == "success":
                    final_output_path = obs_data.get("output_path")
            except (json.JSONDecodeError, AttributeError):
                pass

        return {
            "output": result.get("output", ""),
            "steps": steps,
            "final_output_path": final_output_path,
        }

    def reset_memory(self):
        """Clear conversation history for a fresh session."""
        if self.memory:
            self.memory.clear()
            logger.info("Agent memory cleared")
