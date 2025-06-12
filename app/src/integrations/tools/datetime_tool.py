from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

import isodate
from langchain.tools import BaseTool


class DateTimeTool(BaseTool):
    name: ClassVar[str] = "datetime"
    description: ClassVar[str] = (
        "Return current UTC timestamp or offset timestamp.\n"
        "Args: action('now'|'offset'), offset(ISO-8601 duration when action='offset')."
    )

    def _run(self, action: str = "now", offset: str | None = None, **_: Any) -> str:
        if action == "now":
            return datetime.now(timezone.utc).isoformat()

        if action != "offset" or not offset:
            raise ValueError("Provide offset duration when action == 'offset'")

        delta = isodate.parse_duration(offset)
        if hasattr(delta, "tdelta"):  # Duration → convert months/years
            delta = timedelta(days=delta.days, seconds=delta.tdelta.seconds)

        return (datetime.now(timezone.utc) + delta).isoformat()

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported for DateTimeTool.")
