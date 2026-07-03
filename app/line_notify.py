import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

_TOKEN   = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
_USER_ID = os.getenv("LINE_USER_ID", "")
_URL     = "https://api.line.me/v2/bot/message/push"

log = logging.getLogger(__name__)


def send(text: str) -> None:
    """ส่งข้อความ LINE ไปหา LINE_USER_ID ใน .env"""
    if not _TOKEN or not _USER_ID:
        log.warning("LINE credentials not set — skipping notification")
        return
    try:
        resp = requests.post(
            _URL,
            headers={"Authorization": f"Bearer {_TOKEN}"},
            json={"to": _USER_ID, "messages": [{"type": "text", "text": text}]},
            timeout=10,
        )
        if resp.status_code != 200:
            log.error("LINE send failed %s: %s", resp.status_code, resp.text)
    except Exception as exc:
        log.error("LINE send error: %s", exc)
