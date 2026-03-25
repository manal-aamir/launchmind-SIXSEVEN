#!/usr/bin/env python3
"""
Send a test email with SendGrid before integrating into an agent.

Required .env keys:
- SENDGRID_API_KEY
- SENDGRID_FROM_EMAIL (must be a verified sender in SendGrid)
Optional .env key:
- SENDGRID_TO_EMAIL (defaults to SENDGRID_FROM_EMAIL)
"""

from datetime import datetime, timezone
from pathlib import Path

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from python_http_client.exceptions import HTTPError


def load_env_var(env_path: Path, key: str) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'").strip('"')
    return ""


def main() -> None:
    env_path = Path(__file__).resolve().parent / ".env"

    api_key = load_env_var(env_path, "SENDGRID_API_KEY")
    from_email = load_env_var(env_path, "SENDGRID_FROM_EMAIL")
    to_email = load_env_var(env_path, "SENDGRID_TO_EMAIL") or from_email

    if not api_key:
        raise ValueError("SENDGRID_API_KEY is missing in .env")
    if not from_email:
        raise ValueError(
            "SENDGRID_FROM_EMAIL is missing in .env (use your verified sender email)"
        )
    if not to_email:
        raise ValueError("SENDGRID_TO_EMAIL is missing in .env")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject="SendGrid test email from Python",
        plain_text_content=f"Test email sent at {now}.",
        html_content=f"<strong>SendGrid test email sent at {now}</strong>",
    )

    client = SendGridAPIClient(api_key)
    try:
        response = client.send(message)
    except HTTPError as err:
        print("SendGrid request failed.")
        print("status_code:", err.status_code)
        body = err.body.decode("utf-8", errors="ignore") if isinstance(err.body, bytes) else str(err.body)
        print("error_body:", body)
        raise SystemExit(1) from err

    print("Email request sent to SendGrid.")
    print("from_email:", from_email)
    print("to_email:", to_email)
    print("status_code:", response.status_code)

    # SendGrid often returns an empty body for successful sends (202 Accepted).
    if response.body:
        print("response_body:", response.body.decode("utf-8", errors="ignore"))


if __name__ == "__main__":
    main()
