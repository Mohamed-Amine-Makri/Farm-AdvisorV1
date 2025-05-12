from typing import Any, Dict, Optional
from langchain_core.tools import BaseTool

def create_handoff_tool(agent_name: str, description: Optional[str] = None) -> BaseTool:
    """
    Create a tool that allows an agent to hand off control to another agent.
    
    Args:
        agent_name: The name of the agent to hand off to
        description: A description of when to use this tool
        
    Returns:
        A tool that can be used to hand off control
    """
    # Create a local variable for the description that won't be shadowed
    tool_desc = description if description is not None else f"Hand off to {agent_name}"
    
    class HandoffTool(BaseTool):
        name: str = f"handoff_to_{agent_name}"
        description: str = tool_desc  # Use the local variable here
        
        def _run(self, query: str = "") -> Dict[str, Any]:
            """Run the tool."""
            return {"agent": agent_name, "query": query}
            
        async def _arun(self, query: str = "") -> Dict[str, Any]:
            """Run the tool asynchronously."""
            return self._run(query)
    
    return HandoffTool()