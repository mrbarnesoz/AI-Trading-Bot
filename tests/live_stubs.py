from __future__ import annotations

import sys
import types


def install_live_stubs() -> None:
    """Provide lightweight stand-ins for optional live modules during tests."""
    if "live.policy.decision" in sys.modules:
        return

    live_module = types.ModuleType("live")
    policy_module = types.ModuleType("live.policy")
    decision_module = types.ModuleType("live.policy.decision")
    risk_module = types.ModuleType("live.risk")
    trailing_module = types.ModuleType("live.risk.trailing")
    state_module = types.ModuleType("live.state")
    positions_module = types.ModuleType("live.state.positions")

    class _DummyDecision:
        pass

    class _DummyTrailing:
        pass

    class _DummyPositions:
        pass

    decision_module.MarketState = _DummyDecision
    decision_module.Probabilities = _DummyDecision
    trailing_module.TrailingManager = _DummyTrailing
    trailing_module.PositionManager = _DummyTrailing
    positions_module.PositionManager = _DummyPositions

    sys.modules["live"] = live_module
    sys.modules["live.policy"] = policy_module
    sys.modules["live.policy.decision"] = decision_module
    sys.modules["live.risk"] = risk_module
    sys.modules["live.risk.trailing"] = trailing_module
    sys.modules["live.state"] = state_module
    sys.modules["live.state.positions"] = positions_module


__all__ = ["install_live_stubs"]
