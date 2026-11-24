from opendde_harness.tui_rpc.methods.cli_dispatch import _is_dispatch_compatible


def test_removed_compute_commands_are_unavailable():
    assert _is_dispatch_compatible(["compute", "serve"]) is False
    assert _is_dispatch_compatible(["compute", "serve", "--device", "cpu"]) is False


def test_one_shot_commands_are_dispatchable():
    assert _is_dispatch_compatible(["doctor"]) is True
    assert _is_dispatch_compatible(["protein-design", "initial-round", "--help"]) is True


def test_interactive_entry_points_are_blocked():
    assert _is_dispatch_compatible(["tui"]) is False
    assert _is_dispatch_compatible(["onboard"]) is False
    assert _is_dispatch_compatible(["agent"]) is False
