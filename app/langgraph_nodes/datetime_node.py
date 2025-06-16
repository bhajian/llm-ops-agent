from typing import Dict
from datetime import datetime
import pytz


def datetime_node(state: Dict) -> Dict:
    now = datetime.now(pytz.timezone("America/Toronto")).strftime("%Y-%m-%d %H:%M:%S")
    return {**state, "datetime_result": f"The current time in Toronto is {now}"}
