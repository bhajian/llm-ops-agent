# app/tools/datetime_tools.py
from datetime import datetime
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class GetCurrentDateTimeArgs(BaseModel):
    """Input for GetCurrentDateTimeTool. No arguments are needed for this tool."""
    # This class is required by StructuredTool, even if empty.
    pass

def _get_current_datetime() -> str:
    """
    Returns the current date and time in ISO format (YYYY-MM-DD HH:MM:SS).
    """
    # FIX: Use datetime.now() to get the current date and time dynamically
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_current_datetime_tool = StructuredTool.from_function(
    func=_get_current_datetime,
    name="get_current_datetime",
    description="Useful for when you need to know the current date and time. This tool takes no input.",
    args_schema=GetCurrentDateTimeArgs # Link the input schema
)
