"""Console menu: every item is wired, input is re-asked, errors never crash the loop."""

import builtins

import pytest

import main


def _feed(monkeypatch, *answers):
    replies = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(replies))


def test_every_menu_item_has_an_action():
    assert set(main.MENU) - {"0"} == set(main.ACTIONS)
    assert all(callable(action) for action in main.ACTIONS.values())


def test_ask_int_reasks_until_valid(monkeypatch, capsys):
    _feed(monkeypatch, "abc", "500", "7")
    assert main.ask_int("Slope", 12, 0, 90) == 7
    out = capsys.readouterr().out
    assert "whole number" in out
    assert "between 0 and 90" in out


def test_ask_int_default_on_enter(monkeypatch):
    _feed(monkeypatch, "")
    assert main.ask_int("Slope", 12, 0, 90) == 12


@pytest.mark.parametrize("error, message", [
    (FileNotFoundError("no parquet"), "[missing file] no parquet"),
    (ValueError("bad slope"), "[invalid input] bad slope"),
])
def test_errors_are_reported_and_the_menu_continues(monkeypatch, capsys, error, message):
    def failing():
        raise error

    monkeypatch.setitem(main.ACTIONS, "11", failing)
    monkeypatch.setattr(main.config, "ensure_dirs", lambda: None)
    monkeypatch.setattr(main.sys, "argv", ["main.py"])
    _feed(monkeypatch, "11", "42", "0")
    main.main()
    out = capsys.readouterr().out
    assert message in out
    assert "Unknown option" in out
    assert out.rstrip().endswith("Bye!")
