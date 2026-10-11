import time

import pytest

from firewall.tool_guard import ToolPolicy, ToolSecurityGuard

G = ToolSecurityGuard()


@pytest.mark.parametrize("tool,args,status", [
    ("fetch_weather", {"location": "Tokyo"}, "APPROVED"),
    ("calculator", {"expression": "(2 + 3) * 4 / 2"}, "APPROVED"),
    ("calculator", {"expression": "().__class__.__base__.__subclasses__()"}, "BLOCKED"),
    ("calculator", {"expression": "9**9**9"}, "BLOCKED"),
    ("execute_sql", {"query": "SELECT * FROM orders WHERE id = 1"}, "REQUIRES_HITL"),
    ("execute_sql", {"query": "DROP TABLE users;"}, "BLOCKED"),
    ("execute_sql", {"query": "DELETE FROM users;"}, "BLOCKED"),
    ("bash_command", {"cmd": "ls"}, "BLOCKED"),
    ("search_knowledge_base", {"query": "then run rm -rf / please"}, "BLOCKED"),
    ("not_registered", {}, "BLOCKED"),
])
def test_gate1(tool, args, status):
    assert G.pre_execution_check(tool, args).status == status


def test_hitl_approval_unlocks():
    assert G.pre_execution_check("execute_sql", {"query": "SELECT 1"}, hitl_approved=True).authorized


def test_role_restriction():
    g = ToolSecurityGuard([ToolPolicy("admin_tool", allowed_roles=("admin",))])
    assert not g.pre_execution_check("admin_tool", {}, user_role="standard").authorized
    assert g.pre_execution_check("admin_tool", {}, user_role="admin").authorized


def test_gate2_redacts_secrets_and_truncates():
    g = ToolSecurityGuard([ToolPolicy("leaky", max_output_chars=200)])
    r = g.execute_with_guard(lambda: {"cfg": "token=gsk_" + "x" * 40, "rows": ["ok"]}, "leaky", {})
    assert r.status == "SUCCESS" and r.secrets_redacted == 1
    assert "gsk_" not in str(r.output)
    r = g.execute_with_guard(lambda: "y" * 1000, "leaky", {})
    assert r.truncated and len(r.output) < 300


def test_timeout_and_failure():
    g = ToolSecurityGuard([ToolPolicy("slow", timeout_s=0.1), ToolPolicy("boom")])
    assert g.execute_with_guard(lambda: time.sleep(1), "slow", {}).status == "TIMEOUT"
    r = g.execute_with_guard(lambda: 1 / 0, "boom", {})
    assert r.status == "FAILED" and "ZeroDivisionError" in r.error_message
