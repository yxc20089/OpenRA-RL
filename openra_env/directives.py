"""Strategic directives system for human-as-high-command gameplay.

Enables humans to provide high-level strategic guidance to AI agents.
Directives are injected into the system prompt and accessible via MCP tools.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class DirectiveType(Enum):
    """Type of strategic directive."""

    PREGAME_STRATEGY = "pregame_strategy"
    STANDING_ORDER = "standing_order"
    MIDGAME_ADJUSTMENT = "midgame_adjustment"


@dataclass
class Directive:
    """A single strategic directive from high command."""

    id: int
    type: DirectiveType
    text: str
    acknowledged: bool = False
    created_tick: int = 0
    priority: int = 1  # 1-5, higher = more important (not used in MVP)

    def __str__(self) -> str:
        """Format directive for display."""
        ack_marker = "✓" if self.acknowledged else "NEW"
        return f"[ID:{self.id}] [{ack_marker}] {self.text}"


class DirectivesManager:
    """Manages strategic directives from high command.

    Loads directives from config, formats them for system prompts,
    tracks acknowledgments, and provides status information for MCP tools.
    """

    def __init__(self, config):
        """Initialize DirectivesManager from config.

        Parameters
        ----------
        config : DirectivesConfig
            Directives configuration from OpenRARLConfig.
        """
        self.config = config
        self.directives: list[Directive] = []
        self._next_id = 1
        self._load_directives()

    def _load_directives(self) -> None:
        """Load directives from config and optional external file."""
        # Load from external YAML file if specified
        if self.config.directives_file:
            file_path = Path(self.config.directives_file).expanduser()
            if file_path.exists():
                with open(file_path, encoding="utf-8") as f:
                    file_config = yaml.safe_load(f) or {}

                # Merge file config with inline config (inline takes precedence)
                pregame = self.config.pregame_strategy or file_config.get("pregame_strategy", "")
                standing = self.config.standing_orders or file_config.get("standing_orders", [])
                midgame = self.config.midgame_adjustments or file_config.get("midgame_adjustments", [])
            else:
                pregame = self.config.pregame_strategy
                standing = self.config.standing_orders
                midgame = self.config.midgame_adjustments
        else:
            pregame = self.config.pregame_strategy
            standing = self.config.standing_orders
            midgame = self.config.midgame_adjustments

        # Create directives from config
        if pregame:
            self.directives.append(
                Directive(
                    id=self._next_id,
                    type=DirectiveType.PREGAME_STRATEGY,
                    text=pregame,
                )
            )
            self._next_id += 1

        for order in standing:
            self.directives.append(
                Directive(
                    id=self._next_id,
                    type=DirectiveType.STANDING_ORDER,
                    text=order,
                )
            )
            self._next_id += 1

        for adjustment in midgame:
            self.directives.append(
                Directive(
                    id=self._next_id,
                    type=DirectiveType.MIDGAME_ADJUSTMENT,
                    text=adjustment,
                )
            )
            self._next_id += 1

    def get_all_directives(self) -> list[Directive]:
        """Get all active directives.

        Returns
        -------
        list[Directive]
            List of all directives, regardless of acknowledgment status.
        """
        return self.directives.copy()

    def get_directive_by_id(self, directive_id: int) -> Optional[Directive]:
        """Get a specific directive by ID.

        Parameters
        ----------
        directive_id : int
            The ID of the directive to retrieve.

        Returns
        -------
        Directive or None
            The directive if found, None otherwise.
        """
        for directive in self.directives:
            if directive.id == directive_id:
                return directive
        return None

    def acknowledge_directive(self, directive_id: int) -> bool:
        """Mark a directive as acknowledged by the agent.

        Parameters
        ----------
        directive_id : int
            The ID of the directive to acknowledge.

        Returns
        -------
        bool
            True if directive was found and acknowledged, False otherwise.
        """
        directive = self.get_directive_by_id(directive_id)
        if directive is None:
            return False
        directive.acknowledged = True
        return True

    def get_unacknowledged_count(self) -> int:
        """Get count of directives that haven't been acknowledged.

        Returns
        -------
        int
            Number of unacknowledged directives.
        """
        return sum(1 for d in self.directives if not d.acknowledged)

    def format_for_system_prompt(self) -> str:
        """Format all directives for injection into system prompt.

        Returns
        -------
        str
            Formatted directive section for system prompt, or empty string if no directives.
        """
        if not self.directives:
            return ""

        lines = ["## STRATEGIC DIRECTIVES FROM COMMAND", ""]

        # Group by type
        pregame = [d for d in self.directives if d.type == DirectiveType.PREGAME_STRATEGY]
        standing = [d for d in self.directives if d.type == DirectiveType.STANDING_ORDER]
        midgame = [d for d in self.directives if d.type == DirectiveType.MIDGAME_ADJUSTMENT]

        # Pregame strategy (always at top, no numbering)
        if pregame:
            lines.append("PREGAME STRATEGY:")
            for directive in pregame:
                lines.append(f"  {directive.text}")
            lines.append("")

        # Standing orders
        if standing:
            lines.append("STANDING ORDERS:")
            for directive in standing:
                marker = "✓" if directive.acknowledged else "NEW"
                lines.append(f"  {directive.id}. [{marker}] {directive.text}")
            lines.append("")

        # Midgame adjustments
        if midgame:
            lines.append("TACTICAL ADJUSTMENTS:")
            for directive in midgame:
                marker = "✓" if directive.acknowledged else "URGENT"
                lines.append(f"  {directive.id}. [{marker}] {directive.text}")
            lines.append("")

        # Add usage instructions if acknowledgment is required
        if self.config.acknowledgment_required and self.get_unacknowledged_count() > 0:
            lines.append("Use check_directives() to review all orders.")
            lines.append("Use acknowledge_directive(id) to confirm receipt and execution.")

        return "\n".join(lines)

    def get_status_summary(self) -> dict:
        """Get a summary of directive status for MCP tools.

        Returns
        -------
        dict
            Summary with counts and directive details.
        """
        total = len(self.directives)
        acknowledged = sum(1 for d in self.directives if d.acknowledged)
        unacknowledged = total - acknowledged

        return {
            "total": total,
            "acknowledged": acknowledged,
            "unacknowledged": unacknowledged,
            "directives": [
                {
                    "id": d.id,
                    "type": d.type.value,
                    "text": d.text,
                    "acknowledged": d.acknowledged,
                }
                for d in self.directives
            ],
        }

    def format_for_mcp_tool(self) -> str:
        """Format directives for display in MCP tool responses.

        Returns
        -------
        str
            Human-readable formatted directive list.
        """
        if not self.directives:
            return "No active directives from high command."

        lines = ["=== STRATEGIC DIRECTIVES FROM COMMAND ===", ""]

        # Group by type
        pregame = [d for d in self.directives if d.type == DirectiveType.PREGAME_STRATEGY]
        standing = [d for d in self.directives if d.type == DirectiveType.STANDING_ORDER]
        midgame = [d for d in self.directives if d.type == DirectiveType.MIDGAME_ADJUSTMENT]

        if pregame:
            lines.append("PREGAME STRATEGY:")
            for directive in pregame:
                ack = " [ACKNOWLEDGED]" if directive.acknowledged else ""
                lines.append(f"  {directive.text}{ack}")
            lines.append("")

        if standing:
            lines.append("STANDING ORDERS:")
            for directive in standing:
                ack = " [✓]" if directive.acknowledged else " [NEW]"
                lines.append(f"  {directive.id}. {directive.text}{ack}")
            lines.append("")

        if midgame:
            lines.append("TACTICAL ADJUSTMENTS:")
            for directive in midgame:
                ack = " [✓]" if directive.acknowledged else " [URGENT]"
                lines.append(f"  {directive.id}. {directive.text}{ack}")
            lines.append("")

        # Add statistics
        total = len(self.directives)
        acknowledged = sum(1 for d in self.directives if d.acknowledged)
        unacknowledged = total - acknowledged

        lines.append(f"Status: {acknowledged}/{total} directives acknowledged")
        if unacknowledged > 0:
            lines.append(f"⚠ {unacknowledged} directive(s) awaiting acknowledgment")

        return "\n".join(lines)
