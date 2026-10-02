"""A next step must name a tool the client has, with the action it takes.

`handover.suggested_next` named `{"tool": "read_model_report", "server": "ml_medium"}`; a client has
`ml_predict(action="read_model_report")`, and `ml_medium` has not been an endpoint since the tool
surface was trimmed. The domain dispatcher knows both vocabularies, so it rewrites the pointers.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from servers.ml_domain.server import mcp as domain
from shared.domain_tools import point_at_domains


def _call(tool: str, action: str, **args):
    return asyncio.run(domain._tool_manager._tools[tool].run({"action": action, "args": args}))


def _resolves(pointer: dict) -> bool:
    return (
        pointer["tool"] in domain._tool_manager._tools
        and pointer.get("action")
        in (domain._tool_manager._tools[pointer["tool"]].parameters["properties"]["action"]["enum"])
    )


@pytest.fixture
def table(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("MCP_OUTPUT_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    path = tmp_path / "t.csv"
    pd.DataFrame({"a": rng.normal(0, 1, 200), "y": rng.integers(0, 2, 200)}).to_csv(path, index=False)
    return str(path)


class TestTheRewrite:
    ROUTE = {"read_model_report": "ml_predict"}

    def test_a_pointer_names_the_domain_and_the_action(self):
        result = {"handover": {"suggested_next": [{"tool": "read_model_report", "server": "ml_medium"}]}}
        point = point_at_domains(result, self.ROUTE, "ml")["handover"]["suggested_next"][0]
        assert point == {"tool": "ml_predict", "action": "read_model_report", "server": "ml"}

    def test_an_actions_args_survive(self):
        result = {"insights": [{"action": {"tool": "read_model_report", "args": {"model_path": "/m.pkl"}}}]}
        act = point_at_domains(result, self.ROUTE, "ml")["insights"][0]["action"]
        assert act == {"tool": "ml_predict", "action": "read_model_report", "args": {"model_path": "/m.pkl"}}

    def test_a_hint_names_the_call_that_works(self):
        assert point_at_domains({"hint": "Run read_model_report() next."}, self.ROUTE, "ml")["hint"] == (
            "Run ml_predict(action='read_model_report') next."
        )

    def test_a_tool_this_server_does_not_have_is_left_alone(self):
        result = {"handover": {"suggested_next": [{"tool": "apply_patch", "server": "MCP_Data_Analyst"}]}}
        assert point_at_domains(result, self.ROUTE, "ml")["handover"]["suggested_next"][0]["tool"] == "apply_patch"


class TestEveryPointerResolves:
    @pytest.mark.parametrize(
        ("tool", "action", "extra"),
        [
            ("ml_data", "check_data_quality", {"target_column": "y"}),
            ("ml_train", "train_classifier", {"target_column": "y", "model": "dtc"}),
        ],
    )
    def test_what_an_answer_says_to_do_next_can_be_called(self, table, tool, action, extra):
        pointers = _call(tool, action, file_path=table, **extra)["handover"]["suggested_next"]
        assert pointers
        assert all(_resolves(p) for p in pointers), pointers
