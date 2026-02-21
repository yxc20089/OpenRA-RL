"""Shared test utilities for FastMCP tool access across versions.

FastMCP 3.0 (released 2026-02-19) changed internal APIs — the
``_tool_manager._tools`` dict no longer exists.  This module provides
version-agnostic helpers **and** a pytest autouse fixture that patches
``mcp._tool_manager._tools`` back in so existing tests work unmodified.
"""

import types
import pytest


# ── Version-agnostic tool access helpers ──────────────────────────────────────


def _find_tools_dict(mcp) -> dict | None:
    """Probe FastMCP internals to locate the canonical tool registry.

    Returns the raw dict[str, ToolObj] if found, else ``None``.
    """
    # FastMCP 2.x path
    if hasattr(mcp, "_tool_manager"):
        tm = mcp._tool_manager
        if hasattr(tm, "_tools") and isinstance(tm._tools, dict):
            return tm._tools
        if hasattr(tm, "tools") and isinstance(tm.tools, dict):
            return tm.tools

    # FastMCP 3.x: tools stored directly on mcp
    if hasattr(mcp, "_tools") and isinstance(mcp._tools, dict):
        return mcp._tools

    return None


def _extract_fn(tool_obj):
    """Extract the underlying callable from a Tool wrapper object."""
    if hasattr(tool_obj, "fn"):
        return tool_obj.fn
    if callable(tool_obj):
        return tool_obj
    return None


def get_tool_fn(mcp, name):
    """Get a tool's callable function from a FastMCP instance by name.

    Supports FastMCP 2.x and 3.x.  Returns the raw function so it can
    be called directly in tests.
    """
    tools = _find_tools_dict(mcp)
    if tools is not None:
        tool = tools.get(name)
        if tool is not None:
            return _extract_fn(tool)
    return None


def get_tool_names(mcp) -> set:
    """Return the set of registered tool names."""
    tools = _find_tools_dict(mcp)
    return set(tools.keys()) if tools else set()


def get_tool_count(mcp) -> int:
    """Return the number of registered tools."""
    return len(get_tool_names(mcp))


class ToolWrapper:
    """Compatibility wrapper matching FastMCP 2.x Tool interface."""

    def __init__(self, fn):
        self.fn = fn


def get_tool_obj(mcp, name):
    """Get a tool as an object with a ``.fn`` attribute (FastMCP 2.x compat)."""
    fn = get_tool_fn(mcp, name)
    return ToolWrapper(fn) if fn is not None else None


def get_tools_dict(mcp) -> dict:
    """Return dict mapping tool names → ToolWrapper objects.

    Drop-in replacement for ``mcp._tool_manager._tools``.
    """
    names = get_tool_names(mcp)
    result = {}
    for name in names:
        fn = get_tool_fn(mcp, name)
        if fn is not None:
            result[name] = ToolWrapper(fn)
    return result


# ── Autouse fixtures ──────────────────────────────────────────────────────────

# Monkey-patch FastMCP so that mcp._tool_manager._tools works on 3.x
# This is done via a module-level patch applied when conftest is imported.


def _patch_fastmcp():
    """Ensure FastMCP instances expose ``_tool_manager._tools`` on 3.x."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        return  # fastmcp not installed — nothing to patch

    original_tool = getattr(FastMCP, "tool", None)
    if original_tool is None:
        return

    # Check if _tool_manager._tools already works (FastMCP 2.x)
    test_mcp = FastMCP("__patch_test__")
    if hasattr(test_mcp, "_tool_manager") and hasattr(test_mcp._tool_manager, "_tools"):
        if isinstance(test_mcp._tool_manager._tools, dict):
            return  # Already compatible, no patch needed

    # FastMCP 3.x: We need to create a compatibility shim.
    # Override the tool() method to also store tools in a compat dict.
    _compat_registry = {}  # Will be shared per-mcp instance via __dict__

    original_init = FastMCP.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Add compat _tool_manager._tools
        if not hasattr(self, "_tool_manager"):
            self._tool_manager = types.SimpleNamespace()
        if not hasattr(self._tool_manager, "_tools"):
            self._tool_manager._tools = {}

    def patched_tool(self, *args, **kwargs):
        original_decorator = original_tool(self, *args, **kwargs)

        def wrapper(fn):
            result = original_decorator(fn)
            # Also register in our compat dict
            if hasattr(self, "_tool_manager") and hasattr(self._tool_manager, "_tools"):
                self._tool_manager._tools[fn.__name__] = ToolWrapper(fn)
            return result

        return wrapper

    FastMCP.__init__ = patched_init
    FastMCP.tool = patched_tool


# Apply patch at import time (before any tests run)
_patch_fastmcp()
