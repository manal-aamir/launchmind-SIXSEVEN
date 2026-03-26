"""
QA / Reviewer Agent — Agent 5.

Powered by Groq (llama-3.3-70b-versatile) independently of the other agents.
Reviews the Engineer's HTML and Marketing copy, posts inline PR comments,
and returns a structured pass/fail report to the CEO.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.github_client import GitHubClient


class QAAgent:
    agent_name = "qa"

    def __init__(
        self,
        groq_client: GroqClient,
        github_client: GitHubClient,
        dry_run: bool = True,
    ) -> None:
        self.groq = groq_client
        self.github_client = github_client
        self.dry_run = dry_run

    _BASELINE_COMMENTS = [
        "The features section should explicitly mention the Day 1 / Day 7 / Day 14 "
        "reminder escalation schedule — this is InvoiceHound's core differentiator.",
        "Recommend updating CTA to 'Start Chasing Invoices Free' to match brand voice "
        "and reinforce the no-awkward-follow-up positioning.",
    ]

    @staticmethod
    def _extract_added_lines_from_patch(patch: str) -> List[int]:
        added: List[int] = []
        current_new_line = 0

        for raw in patch.splitlines():
            if raw.startswith("@@"):
                match = re.search(r"\+(\d+)(?:,\d+)?", raw)
                if match:
                    current_new_line = int(match.group(1))
                continue

            if raw.startswith("+") and not raw.startswith("+++"):
                if current_new_line > 0:
                    added.append(current_new_line)
                current_new_line += 1
            elif raw.startswith("-") and not raw.startswith("---"):
                continue
            else:
                if current_new_line > 0:
                    current_new_line += 1

        return added

    def _post_two_review_comments(
        self,
        pr_number: int,
        head_sha: str,
        comments: List[str],
    ) -> Dict[str, Any]:
        pr_files = self.github_client.list_pr_files(pr_number)
        index_file = next((f for f in pr_files if f.get("filename") == "index.html"), None)
        patch = str((index_file or {}).get("patch", ""))
        added_lines = self._extract_added_lines_from_patch(patch)

        line1: Optional[int] = added_lines[0] if len(added_lines) >= 1 else None
        line2: Optional[int] = added_lines[1] if len(added_lines) >= 2 else line1

        if not line1 or not line2:
            raise ValueError("No suitable added lines found for inline comments")

        posted: List[Dict[str, Any]] = []
        posted.append(
            self.github_client.create_review_comment_inline(
                pr_number=pr_number,
                commit_id=head_sha,
                path="index.html",
                body=comments[0],
                line=line1,
                side="RIGHT",
            )
        )
        posted.append(
            self.github_client.create_review_comment_inline(
                pr_number=pr_number,
                commit_id=head_sha,
                path="index.html",
                body=comments[1],
                line=line2,
                side="RIGHT",
            )
        )
        return {"ok": True, "mode": "inline", "comments": posted}

    def run(self, outputs: Dict[str, Dict]) -> Tuple[bool, str, List[str], Dict]:
        product = outputs.get("product", {})
        engineer = outputs.get("engineer", {})
        marketing = outputs.get("marketing", {})
        html = str(engineer.get("html", ""))

        html_review = self.groq.review_html(product, html)
        copy_review = self.groq.review_copy(marketing)

        issues: List[str] = []
        issues.extend(list(html_review.get("issues", [])))
        issues.extend(list(copy_review.get("issues", [])))

        html_verdict = str(html_review.get("verdict", "fail")).lower()
        copy_verdict = str(copy_review.get("verdict", "fail")).lower()
        passed = (html_verdict == "pass") and (copy_verdict == "pass") and not issues

        review_receipt: Dict[str, Any] = {"ok": "dry-run", "reason": "dry_run_or_missing_pr"}
        pr_number = engineer.get("pr_number")

        if (not self.dry_run) and pr_number:
            pr = self.github_client.get_pr(int(pr_number))
            head_sha = str(pr.get("head", {}).get("sha", ""))

            raw_comments = list(self._BASELINE_COMMENTS)
            for issue in issues:
                raw_comments.append(self.groq.generate_pr_comment(issue))

            try:
                review_receipt = self._post_two_review_comments(
                    pr_number=int(pr_number),
                    head_sha=head_sha,
                    comments=raw_comments[:2],
                )
            except Exception:
                c1 = self.github_client.create_review_comment(
                    pr_number=int(pr_number),
                    commit_id=head_sha,
                    path="index.html",
                    body=raw_comments[0],
                )
                c2 = self.github_client.create_review_comment(
                    pr_number=int(pr_number),
                    commit_id=head_sha,
                    path="index.html",
                    body=raw_comments[1],
                )
                review_receipt = {"ok": True, "mode": "file", "comments": [c1, c2]}

        notes = "QA passed — HTML and copy approved." if passed else "QA failed — issues found."
        report = {
            "llm": f"groq/{self.groq.model}",
            "html_review": html_review,
            "copy_review": copy_review,
            "review_receipt": review_receipt,
        }
        return passed, notes, issues, report


__all__ = ["QAAgent"]

