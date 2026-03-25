"""GitHub REST client for issue, branch, commit, PR, and review comments."""

import base64
import json
from typing import Any, Dict, List, Optional
from urllib import parse, request


class GitHubClient:
    def __init__(self, token: str, repo: str) -> None:
        self.token = token
        self.repo = repo  # owner/repo
        self.api_base = f"https://api.github.com/repos/{repo}"

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.api_base}{path}",
            data=data,
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "invoicehound-agent",
            },
            method=method,
        )
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}

    def get_repo(self) -> Dict[str, Any]:
        return self._request("GET", "")

    def create_issue(self, title: str, body: str) -> Dict[str, Any]:
        return self._request("POST", "/issues", {"title": title, "body": body})

    def get_ref_sha(self, branch: str) -> str:
        ref = self._request("GET", f"/git/ref/heads/{parse.quote(branch)}")
        return str(ref["object"]["sha"])

    def create_branch(self, new_branch: str, from_branch: str) -> Dict[str, Any]:
        sha = self.get_ref_sha(from_branch)
        return self._request(
            "POST",
            "/git/refs",
            {"ref": f"refs/heads/{new_branch}", "sha": sha},
        )

    def get_file_sha(self, path: str, ref: str) -> str:
        payload = self._request("GET", f"/contents/{parse.quote(path)}?ref={parse.quote(ref)}")
        return str(payload["sha"])

    def upsert_file(self, branch: str, path: str, content_text: str, message: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content_text.encode("utf-8")).decode("utf-8"),
            "branch": branch,
            "committer": {"name": "EngineerAgent", "email": "agent@invoicehound.ai"},
            "author": {"name": "EngineerAgent", "email": "agent@invoicehound.ai"},
        }
        try:
            body["sha"] = self.get_file_sha(path=path, ref=branch)
        except Exception:
            pass
        return self._request("PUT", f"/contents/{parse.quote(path)}", body)

    def create_pr(self, title: str, body: str, head: str, base: str) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/pulls",
            {"title": title, "body": body, "head": head, "base": base},
        )

    def get_pr(self, pr_number: int) -> Dict[str, Any]:
        return self._request("GET", f"/pulls/{pr_number}")

    def create_inline_review_comments(
        self, pr_number: int, comments: List[Dict[str, Any]], body: str = "QA review comments"
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/pulls/{pr_number}/reviews",
            {"body": body, "event": "COMMENT", "comments": comments},
        )

