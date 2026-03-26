"""SendGrid helper for marketing email actions."""

import base64
from typing import Dict, Optional

from python_http_client.exceptions import HTTPError
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Attachment, FileContent, FileName, FileType, Mail


class SendGridClient:
    def __init__(self, api_key: str, from_email: str, to_email: str) -> None:
        self.api_key = api_key
        self.from_email = from_email
        self.to_email = to_email

    def send_email(
        self,
        subject: str,
        plain_text: str,
        html_text: str,
        pdf_bytes: Optional[bytes] = None,
        pdf_filename: str = "invoice.pdf",
    ) -> Dict[str, str]:
        message = Mail(
            from_email=self.from_email,
            to_emails=self.to_email,
            subject=subject,
            plain_text_content=plain_text,
            html_content=html_text,
        )
        if pdf_bytes:
            attachment = Attachment(
                FileContent(base64.b64encode(pdf_bytes).decode()),
                FileName(pdf_filename),
                FileType("application/pdf"),
            )
            message.attachment = attachment
        client = SendGridAPIClient(self.api_key)
        try:
            response = client.send(message)
        except HTTPError as err:
            body = err.body.decode("utf-8", errors="ignore") if isinstance(err.body, bytes) else str(err.body)
            return {"ok": "False", "status_code": str(err.status_code), "error": body}
        return {"ok": "True", "status_code": str(response.status_code), "error": ""}

