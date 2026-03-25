"""SendGrid helper for marketing email actions."""

from typing import Dict

from python_http_client.exceptions import HTTPError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


class SendGridClient:
    def __init__(self, api_key: str, from_email: str, to_email: str) -> None:
        self.api_key = api_key
        self.from_email = from_email
        self.to_email = to_email

    def send_email(self, subject: str, plain_text: str, html_text: str) -> Dict[str, str]:
        message = Mail(
            from_email=self.from_email,
            to_emails=self.to_email,
            subject=subject,
            plain_text_content=plain_text,
            html_content=html_text,
        )
        client = SendGridAPIClient(self.api_key)
        try:
            response = client.send(message)
        except HTTPError as err:
            body = err.body.decode("utf-8", errors="ignore") if isinstance(err.body, bytes) else str(err.body)
            return {"ok": "False", "status_code": str(err.status_code), "error": body}
        return {"ok": "True", "status_code": str(response.status_code), "error": ""}

