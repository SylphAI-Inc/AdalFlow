"""Regression: require_approval=True must not run without a callable callback.

Covers SylphAI-Inc/AdalFlow#501 — previously check_permission allowed on a
falsy approval_callback (None / '' / 0) after only a warning.
"""

from __future__ import annotations

import asyncio

import pytest

from adalflow.apps.permission_manager import ApprovalOutcome, PermissionManager
from adalflow.core.types import Function


TOOL = "delete_everything"


async def _approve(_request):
    return ApprovalOutcome.PROCEED_ONCE


async def _cancel(_request):
    return ApprovalOutcome.CANCEL


@pytest.mark.parametrize(
    "callback",
    [None, "", 0],
    ids=["None", "empty-str", "zero"],
)
def test_require_approval_denies_without_callable_callback(callback):
    permissions = PermissionManager(approval_callback=callback)
    permissions.register_tool(TOOL, require_approval=True)
    func = Function(name=TOOL, args=[], kwargs={})

    allowed, modified, response = asyncio.run(permissions.check_permission(func))

    assert allowed is False
    assert modified is None
    assert response is None


def test_require_approval_still_honors_approving_callback():
    permissions = PermissionManager(approval_callback=_approve)
    permissions.register_tool(TOOL, require_approval=True)
    func = Function(name=TOOL, args=[], kwargs={})

    allowed, modified, _ = asyncio.run(permissions.check_permission(func))

    assert allowed is True
    assert modified is not None


def test_require_approval_still_honors_denying_callback():
    permissions = PermissionManager(approval_callback=_cancel)
    permissions.register_tool(TOOL, require_approval=True)
    func = Function(name=TOOL, args=[], kwargs={})

    allowed, modified, _ = asyncio.run(permissions.check_permission(func))

    assert allowed is False
    assert modified is None


@pytest.mark.parametrize(
    "callback",
    [1, "true"],
    ids=["int-one", "str-true"],
)
def test_require_approval_denies_truthy_non_callable(callback):
    """Truthy non-callables must also fail closed (not rely on await TypeError)."""
    permissions = PermissionManager(approval_callback=callback)
    permissions.register_tool(TOOL, require_approval=True)
    func = Function(name=TOOL, args=[], kwargs={})

    allowed, modified, _ = asyncio.run(permissions.check_permission(func))

    assert allowed is False
    assert modified is None


def test_unregistered_tool_still_runs_without_callback():
    """Default remains: tools not marked require_approval do not need a callback."""
    permissions = PermissionManager(approval_callback=None)
    func = Function(name="safe_read", args=[], kwargs={})

    allowed, modified, _ = asyncio.run(permissions.check_permission(func))

    assert allowed is True
    assert modified is func
