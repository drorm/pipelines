import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class SessionLogger:
    def __init__(
        self,
        base_dir: Path = Path(
            os.getenv("ANTHROPIC_CONFIG_DIR", "~/.anthropic")
        ).expanduser(),
    ):
        self.base_dir = base_dir
        self.logs_dir = base_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = self.logs_dir / "template.md"
        self.current_file: Optional[Path] = None

    def start_session(self, request: str) -> Path:
        timestamp = datetime.now()
        filename = timestamp.strftime("%Y-%m-%d_%H%M%S.md")
        self.current_file = self.logs_dir / filename

        # Generate summary from first request
        summary = self._generate_summary(request)

        # Read template
        template = self.template_path.read_text()

        # Replace placeholders
        log_content = template.replace(
            "# Subject -- short description of the request", f"# {summary}"
        )
        log_content = log_content.replace(
            "$timestamp", timestamp.strftime("%B %d, %Y at %H:%M:%S")
        )

        # Initialize with first exchange
        log_content += "\n## Exchange 1\n\n"
        log_content += "### User\n" + request + "\n\n"
        log_content += "### Assistant\n"  # Empty initially

        # Write to file
        self.current_file.write_text(log_content)
        return self.current_file

    def append_response(self, response: str) -> None:
        """Append an assistant's response to the current exchange"""
        if not self.current_file:
            raise RuntimeError("No active session file")

        content = self.current_file.read_text()
        # Add response under the last Assistant section
        content += response + "\n\n"
        self.current_file.write_text(content)

    def append_user_message(self, message: str) -> None:
        """Start a new exchange with a user message"""
        if not self.current_file:
            raise RuntimeError("No active session file")

        content = self.current_file.read_text()
        # Count existing exchanges
        exchange_count = content.count("## Exchange")

        # Add new exchange section
        content += f"\n## Exchange {exchange_count + 1}\n\n"
        content += "### User\n" + message + "\n\n"
        content += "### Assistant\n"

        self.current_file.write_text(content)

    def _generate_summary(self, request: str) -> str:
        # Simple summary generation - take first line and truncate
        first_line = request.strip().split("\n")[0]
        return first_line[:50] + ("..." if len(first_line) > 50 else "")
