"""token_estimate was a literal, so it described nothing.

CLAUDE.md defines it as `len(str(response)) // 4`. It was hardcoded instead --
**101** occurrences of `"token_estimate": 15` and friends in this repo, the
largest count in the fleet. Measured against the live endpoints before the fix:

    inspect_dataset("/nonexistent/definitely_not_here.csv")
    -> an error carrying the whole path, and a hint naming what to do
       "token_estimate": 15          nowhere near what the response costs

This repo is a good illustration of why a blanket literal survives so long:
some of its responses were already honest. read_receipt returned a computed
258 on a real call, so spot-checking one tool proves nothing about the rest.

Under-reporting is the direction that hurts. A client budgets its context from
this number and admits the response on the strength of it, so an
order-of-magnitude undercount blows the ~12,000-token budget these servers are
designed around. Error responses are the worst case, because their length is
dominated by a variable-length message: a constant is wrong by construction,
and *improving* a message by making it more specific silently makes the lie
bigger.

MCP_Math is the only repo in the fleet that computes it, in
src/engine/formatter.py. This is the same idea applied here as a single choke
point, because 101 hand-edits drift out of step with the responses exactly the
way the literals already had.

These assertions reach each tool through the registry, which is the path a
client's request takes. That matters more than it looks: measure_responses
wraps the registry entry, and the official MCP SDK's @mcp.tool hands back the
plain undecorated function, so calling the module-level name would skip the
wrapper entirely and pass while the thing it guards was switched off. A fix the
suite dispatches around is a fix nobody can hold to account.
"""

from __future__ import annotations

import importlib

import pytest


def tool_fn(mod, name: str):
    """The callable a client actually reaches, via the tool registry.

    Under fastmcp 2.x the module-level name WAS the registry entry, so
    `mod.some_tool.fn` and the client's path were the same object. The official
    MCP SDK's @mcp.tool returns the plain undecorated function instead, so the
    module-level name now bypasses every wrapper installed on the registry --
    measure_responses, contract_errors, sanitize_responses.

    Going through _tools keeps these tests on the path a request takes. A test
    that calls the bare function would pass while the thing it guards was
    switched off, which is the one failure mode these guards exist to prevent.
    """
    return mod.mcp._tool_manager._tools[name].fn


# One cheap, reliably-failing call per server: a path that cannot exist. The
# error message carries a variable-length path, which is precisely the shape a
# constant estimate cannot describe.
MISSING = "/nonexistent/definitely_not_here.csv"

CASES = [
    ("servers.ml_basic.server", "inspect_dataset", {"file_path": MISSING}),
    ("servers.ml_basic.server", "read_rows", {"file_path": MISSING, "start": 0, "end": 5}),
    ("servers.ml_medium.server", "check_data_quality", {"file_path": MISSING}),
    ("servers.ml_medium.server", "read_receipt", {"file_path": MISSING}),
    ("servers.ml_advanced.server", "read_model_report", {"model_path": MISSING}),
]


def measured(response: dict) -> int:
    """What the contract says the estimate should be for this response.

    token_estimate is dropped before measuring, so the number describes the
    payload rather than partly describing itself. recount() sets the key last,
    so removing it here leaves the other keys in their original order and
    str() renders the same bytes.
    """
    return len(str({k: v for k, v in response.items() if k != "token_estimate"})) // 4


def call(module: str, tool: str, kwargs: dict) -> dict:
    mod = importlib.import_module(module)
    return tool_fn(mod, tool)(**kwargs)


@pytest.mark.parametrize("module,tool,kwargs", CASES, ids=[f"{m.split('.')[1]}.{t}" for m, t, _ in CASES])
class TestTheEstimateIsMeasuredNotTypedIn:
    def test_it_matches_the_response_it_describes(self, module: str, tool: str, kwargs: dict) -> None:
        r = call(module, tool, kwargs)
        assert isinstance(r, dict), f"{tool} did not return a dict"
        assert r["token_estimate"] == measured(r), (
            f"{tool} reported {r['token_estimate']}, response measures {measured(r)}"
        )

    def test_it_is_not_one_of_the_stock_literals(self, module: str, tool: str, kwargs: dict) -> None:
        """15, 20, 25, 30 and 40 were the hardcoded values across 101 sites.

        A stock value is only a failure when it disagrees with the measurement;
        a genuinely 60-character response really does estimate 15.
        """
        r = call(module, tool, kwargs)
        if r["token_estimate"] in (15, 20, 25, 30, 40):
            assert r["token_estimate"] == measured(r), (
                f"{tool} returned the stock literal {r['token_estimate']}; the response measures {measured(r)}"
            )


class TestTheDirectionThatMatters:
    """Equality alone would be satisfied by any consistently wrong number."""

    def test_a_long_error_is_nowhere_near_the_literal(self) -> None:
        r = call("servers.ml_basic.server", "inspect_dataset", {"file_path": MISSING})
        assert r["token_estimate"] > 15, r
        assert r["token_estimate"] == measured(r)


class TestRecountItself:
    def test_it_measures_without_counting_its_own_field(self) -> None:
        from shared.token_estimate import recount

        r = recount({"success": True, "note": "x" * 400, "token_estimate": 15})
        assert r["token_estimate"] == measured(r)
        assert r["token_estimate"] > 90

    def test_a_stale_literal_is_replaced_not_kept(self) -> None:
        from shared.token_estimate import recount

        assert recount({"a": 1, "token_estimate": 9999})["token_estimate"] != 9999

    def test_a_response_with_no_estimate_gets_one(self) -> None:
        from shared.token_estimate import recount

        assert "token_estimate" in recount({"success": True})

    def test_a_non_dict_is_left_alone(self) -> None:
        """The return contract is enforced elsewhere; this must not raise."""
        from shared.token_estimate import recount

        assert recount("not a dict") == "not a dict"
        assert recount(None) is None

    def test_installing_twice_does_not_double_wrap(self) -> None:
        """measure_responses is safe to call again on an already-wrapped server."""
        from mcp.server.fastmcp import FastMCP

        from shared.token_estimate import measure_responses

        m = FastMCP("probe")

        @m.tool()
        def sample(x: int) -> dict:
            """doc"""
            return {"x": x, "token_estimate": 15}

        measure_responses(m)
        fn = m._tool_manager._tools["sample"].fn
        once = fn(x=1)["token_estimate"]
        measure_responses(m)
        assert m._tool_manager._tools["sample"].fn(x=1)["token_estimate"] == once
