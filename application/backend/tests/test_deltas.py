"""Property 39: Delta wrapper equivalence (Req 12.1, 12.2)."""
from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from db import delta_engine
from genai import diet_planner
from strategies import visit_dicts


@given(visit_dicts(), st.one_of(st.none(), visit_dicts()))
def test_wrapper_matches_delta_engine(current, previous):
    assert diet_planner.compute_visit_deltas(current, previous) == delta_engine.compute_visit_deltas(previous, current)
