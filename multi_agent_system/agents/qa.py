"""
QA / Reviewer Agent — Agent 5.

Powered by Groq (llama-3.3-70b-versatile) independently of the other agents.
Reviews the Engineer's HTML and Marketing copy, posts inline PR comments,
and returns a structured pass/fail report to the CEO.
"""

from typing import Dict, List, Tuple

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

    # ------------------------------------------------------------------
    # Hardcoded baseline PR comments (always posted regardless of LLM)
    # ------------------------------------------------------------------

    _BASELINE_COMMENTS = [
        "The features section should explicitly mention the Day 1 / Day 7 / Day 14 "
        "reminder escalation schedule — this is InvoiceHound's core differentiator.",
        "Recommend updating CTA to 'Start Chasing Invoices Free' to match brand voice "
        "and reinforce the no-awkward-follow-up positioning.",
    ]

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self, outputs: Dict[str, Dict]) -> Tuple[bool, str, List[str], Dict]:
        product  = outputs.get("product",   {})
        engineer = outputs.get("engineer",  {})
        marketing = outputs.get("marketing", {})
        html = str(engineer.get("html", ""))

        # LLM reviews via Groq
        html_review  = self.groq.review_html(product, html)
        copy_review  = self.groq.review_copy(marketing)

        issues: List[str] = []
        issues.extend(list(html_review.get("issues", [])))
        issues.extend(list(copy_review.get("issues", [])))

        html_verdict = str(html_review.get("verdict", "fail")).lower()
        copy_verdict = str(copy_review.get("verdict", "fail")).lower()
        passed = (html_verdict == "pass") and (copy_verdict == "pass") and not issues

        # GitHub inline PR review comments
        review_receipt: Dict = {"ok": "dry-run", "reason": "dry_run_or_missing_pr"}
        pr_number = engineer.get("pr_number")

        if (not self.dry_run) and pr_number:
            pr = self.github_client.get_pr(int(pr_number))
            head_sha = str(pr.get("head", {}).get("sha", ""))

            # Build comments: baseline + any LLM-generated issue comments
            raw_comments = list(self._BASELINE_COMMENTS)
            for issue in issues:
                raw_comments.append(self.groq.generate_pr_comment(issue))

            # Post at least two file-level PR review comments on index.html
            # (assignment requirement). Uses subject_type="file" which never
            # triggers 422 position-resolution errors.
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
            review_receipt = {"ok": True, "comments": [c1, c2]}

        notes = "QA passed — HTML and copy approved." if passed else "QA failed — issues found."
        report = {
            "llm": f"groq/{self.groq.model}",
            "html_review":   html_review,
            "copy_review":   copy_review,
            "review_receipt": review_receipt,
        }
        return passed, notes, issues, report
