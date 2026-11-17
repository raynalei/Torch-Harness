"""What a tool result costs the turn is bounded where it enters the turn.

Measured before this: one iteration with three 250KB status dumps made a
360k-token request on a model whose table says 272k. Two rules now hold:
each result is cut to ``_TOOL_RESULT_MAX_CHARS`` before the model sees it
(and the record shows exactly what the model saw), and a prompt the window
cannot hold has its older tool bodies elided before the call, not only after
an overflow error.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from opendde_harness.agent.loop import AgentLoop
from opendde_harness.agent.loop.factory import AgentLoopSettings
from opendde_harness.agent.loop.tool_batch import TOOL_PREVIEW_MAX_CHARS
from opendde_harness.agent.tools.base import Tool, ToolResult
from opendde_harness.config.schema import ProvidersConfig
from opendde_harness.providers import messages as msg
from opendde_harness.providers.base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    compaction_boundary,
    wire_history,
)
from opendde_harness.providers.messages import text_of
from opendde_harness.session.manager import SessionManager
from opendde_harness.spine import ChatType, Origin, Source, TurnRequest
from opendde_harness.token_wise.base import TokenStrategy
from opendde_harness.utils.helpers import estimate_prompt_tokens
from tests import _messages as build
from tests._config import config as build_config
from tests._config import declared

MODEL = "primary/model"


def _windows(**models: int | tuple[int, int]) -> ProvidersConfig:
    """The provider entry declaring these models' windows, and nothing else.

    A window is declared per model, as a row in the provider's own entry, so a
    turn routed to another model is sized by that model's own row.
    ``model=8000`` declares the window; ``model=(8000, 1000)`` declares the
    reply's own ceiling beside it. ``primary`` is a name pi does not ship, so
    the entry is a provider this config declares -- an address and the protocol
    it speaks, which :func:`declared` pairs for us.
    """
    rows: list[dict[str, object]] = []
    for name, value in models.items():
        window, ceiling = value if isinstance(value, tuple) else (value, None)
        row: dict[str, object] = {"id": name, "contextWindow": window}
        if ceiling is not None:
            row["maxTokens"] = ceiling
        rows.append(row)
    return build_config(declared("primary", models=rows), model=MODEL).providers


class Blob(Tool):
    """Returns a result of a chosen size."""

    def __init__(self, size: int) -> None:
        self.size = size

    @property
    def name(self) -> str:
        return "blob"

    @property
    def description(self) -> str:
        return "a big result"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        # Varied words, not one repeated character: a run of one byte tokenizes
        # to almost nothing, and the fit test needs a realistic count.
        words = (f"word{i} " for i in range(self.size))
        out = ""
        for w in words:
            if len(out) + len(w) > self.size:
                break
            out += w
        return out.ljust(self.size, "x")


class CallsBlob(LLMProvider):
    """First call asks for ``calls`` blobs at once, the second answers."""

    def __init__(self, calls: int = 1) -> None:
        super().__init__()
        self.calls = calls
        self.prompts: list[list[dict]] = []

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.prompts.append([dict(m) for m in messages])
        if len(self.prompts) == 1:
            requests = [ToolCallRequest(id=f"c{i}", name="blob", arguments={}) for i in range(self.calls)]
            return LLMResponse(content="", tool_calls=requests, finish_reason="tool_calls")
        return LLMResponse(content="done", finish_reason="stop")

    def get_default_model(self) -> str:
        return MODEL


def _request() -> TurnRequest:
    return TurnRequest(
        origin=Origin.USER,
        source=Source(channel="cli", chat_id="t", sender_id="user", chat_type=ChatType.DM),
        text="go",
        conversation="cli:t",
    )


async def _emit(event) -> None:
    pass


def _loop(tmp_path, provider: LLMProvider, size: int, **settings) -> AgentLoop:
    loop = AgentLoop(provider, tmp_path, AgentLoopSettings(model=MODEL, **settings))
    loop.tools.register(Blob(size))
    return loop


def _tool_contents(prompt: list[dict]) -> list[str]:
    return [text_of(m) for m in prompt if m.get("role") == "toolResult"]


async def test_a_tool_result_is_cut_before_the_model_sees_it_and_the_record_matches(tmp_path) -> None:
    provider = CallsBlob()
    loop = _loop(tmp_path, provider, size=100_000)

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    (seen,) = _tool_contents(provider.prompts[1])
    body_limit = AgentLoop._TOOL_RESULT_MAX_CHARS
    body = seen.split("\n", 1)[1]
    assert body.startswith("word0 word1 ") and body.index("\n\n[tool result truncated") == body_limit
    assert f"the first {body_limit:,} of 100,000 characters are shown" in seen
    assert seen.rstrip().endswith("]") and "[END UNTRUSTED blob" in seen, "the fence survives the cut"
    assert len(seen) < body_limit + 600

    stored = _tool_contents(loop.sessions.get_or_create("cli:t").messages)
    assert stored == [seen], "the record holds what the model was shown"
    await loop.close_mcp()


async def test_a_short_result_is_left_alone(tmp_path) -> None:
    provider = CallsBlob()
    loop = _loop(tmp_path, provider, size=500)
    await loop.run_turn(_request(), _emit, lambda: [], stream=False)
    (seen,) = _tool_contents(provider.prompts[1])
    assert len(seen.split("\n")[1]) == 500 and "truncated" not in seen
    await loop.close_mcp()


async def test_the_row_is_sent_a_kilobyte_of_the_result_and_told_when_that_is_not_all_of_it(tmp_path) -> None:
    """The preview is what ctrl+o expands to, so it holds a useful amount (it
    held 200 characters, and the row said "truncated" on nearly every call).
    ``truncated`` says the preview is the head of a longer result."""
    from opendde_harness.spine import ToolEvent, ToolPhase
    from tests._gate import Events

    def completes(events: Events) -> list[ToolEvent]:
        return [event for event in events.of(ToolEvent) if event.phase == ToolPhase.COMPLETE]

    whole = Events()
    loop = _loop(tmp_path, CallsBlob(), size=500)
    await loop.run_turn(_request(), whole, lambda: [], stream=False)
    await loop.close_mcp()

    cut = Events()
    loop = _loop(tmp_path, CallsBlob(), size=TOOL_PREVIEW_MAX_CHARS + 1)
    await loop.run_turn(_request(), cut, lambda: [], stream=False)
    await loop.close_mcp()

    (kept,) = completes(whole)
    (capped,) = completes(cut)
    assert (kept.truncated, len(kept.result_preview)) == (False, 500), "the whole result, no note"
    assert (capped.truncated, len(capped.result_preview)) == (True, TOOL_PREVIEW_MAX_CHARS)


@pytest.mark.parametrize("window", [None, 30_000])
async def test_older_tool_bodies_are_elided_before_a_call_the_window_cannot_hold(tmp_path, window) -> None:
    """Six 20k-char results (a few thousand tokens each) against a 30k window
    whose reply ceiling is 16k leave the prompt over its budget, so the older
    results are elided before the call. An unknown window elides nothing."""
    providers = ProvidersConfig() if window is None else _windows(model=window)
    provider = CallsBlob(calls=6)
    loop = _loop(tmp_path, provider, size=20_000, providers=providers)

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    contents = _tool_contents(provider.prompts[1])
    elided = [c for c in contents if c == "[earlier tool output elided to fit the context window]"]
    if window is None:
        assert elided == []
    else:
        assert len(elided) == 6 - AgentLoop._SHRINK_KEEP_RECENT_TOOL_RESULTS
        assert all("word0 " in c for c in contents[-AgentLoop._SHRINK_KEEP_RECENT_TOOL_RESULTS :])
    await loop.close_mcp()


class WithheldTools(TokenStrategy):
    """What ToolSearchStrategy does to a large catalog: the request carries no schemas."""

    @property
    def name(self) -> str:
        return "withheld"

    async def before_llm_call(self, messages, tools, model):
        return messages, [], model


class Verbose(Tool):
    def __init__(self, n: int) -> None:
        self.n = n

    @property
    def name(self) -> str:
        return f"verbose_{self.n}"

    @property
    def description(self) -> str:
        return " ".join(f"detail{i}" for i in range(400))

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return "never called"


async def test_the_prompt_is_fitted_against_what_is_sent_not_the_withheld_catalog(tmp_path) -> None:
    """Measured before this: a catalog the strategy withholds pushed the
    estimate over budget and elided history the actual request had room for."""
    provider = CallsBlob(calls=4)
    loop = _loop(tmp_path, provider, size=200, providers=_windows(model=30_000))
    for n in range(40):
        loop.tools.register(Verbose(n))
    assert estimate_prompt_tokens([], loop.tools.get_definitions()) > 30_000 - 16_384, "the catalog alone is over"
    loop.strategies.register(WithheldTools())

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    contents = _tool_contents(provider.prompts[1])
    assert len(contents) == 4 and all("word0 " in c for c in contents), "nothing elided: the request fit"
    await loop.close_mcp()


class Picture(Tool):
    """A multimodal result: a long text part beside an image."""

    def __init__(self, size: int) -> None:
        self.size = size

    @property
    def name(self) -> str:
        return "picture"

    @property
    def description(self) -> str:
        return "renders"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        text = ("word " * (self.size // 5))[: self.size]
        blocks = [msg.text_block(text), msg.image_block("AAAA", "image/png")]
        return ToolResult(text, "a picture", blocks=blocks)


class CallsPicture(CallsBlob):
    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.prompts.append([dict(m) for m in messages])
        if len(self.prompts) == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c0", name="picture", arguments={})],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="done", finish_reason="stop")


async def test_the_text_of_a_multimodal_result_is_under_the_same_cap(tmp_path) -> None:
    """On an image-capable route the blocks replace the text, so the cap has
    to reach the text inside them; the picture passes through."""
    provider = CallsPicture()
    loop = AgentLoop(provider, tmp_path, AgentLoopSettings(model=MODEL))
    loop.tools.register(Picture(100_000))
    loop._supports_vision = lambda model=None: True

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    (result,) = [m for m in provider.prompts[1] if m.get("role") == "toolResult"]
    blocks = result["content"]
    texts = [b["text"] for b in blocks if b["type"] == "text"]
    assert len(texts) == 1 and len(texts[0]) < AgentLoop._TOOL_RESULT_MAX_CHARS + 600
    assert "the first 16,000 of 100,000 characters are shown" in texts[0]
    assert [b["type"] for b in blocks] == ["text", "image"]
    await loop.close_mcp()


class RoutesToBig(TokenStrategy):
    """A strategy that sends the call to another model, as the contract allows."""

    @property
    def name(self) -> str:
        return "routes"

    async def before_llm_call(self, messages, tools, model):
        return messages, tools, "primary/big"


async def test_the_prompt_is_fitted_against_the_model_the_call_goes_to(tmp_path) -> None:
    """The turn's model has a small window; the strategy routes the call to a
    model with a large one, and the large one's budget is what applies."""
    provider = CallsBlob(calls=6)
    loop = _loop(tmp_path, provider, size=15_000, providers=_windows(model=30_000, big=400_000))
    loop.strategies.register(RoutesToBig())

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    contents = _tool_contents(provider.prompts[1])
    assert len(contents) == 6 and all("word0 " in c for c in contents), "nothing elided: the big model has room"
    await loop.close_mcp()


class ReasonsThenCalls(CallsBlob):
    """A Codex-style turn: every answer carries an encrypted reasoning replay."""

    async def chat(self, messages, tools=None, model=None, **kwargs):
        response = await super().chat(messages, tools, model, **kwargs)
        response.thinking_blocks = [
            {
                "type": "opendde_reasoning",
                "provider": "openai-codex",
                "thinking": "thinking",
                "items": [{"type": "reasoning", "id": f"rs_{len(self.prompts)}", "encrypted_content": "Z" * 200_000}],
                "order": [],
                "messages": {},
            }
        ]
        return response


async def test_encrypted_reasoning_in_the_replay_does_not_make_the_fitter_elide(tmp_path) -> None:
    """The regression behind a 52% cache hit rate: the replayed blobs were
    counted as prompt text, the estimate blew past the window, and every
    call elided one more tool result, rewriting the cached prefix."""
    provider = ReasonsThenCalls(calls=6)
    loop = _loop(tmp_path, provider, size=2_000, providers=_windows(model=60_000))

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    contents = _tool_contents(provider.prompts[1])
    assert len(contents) == 6 and all("word0 " in c for c in contents), "a 12k-char prompt fits a 60k window"


# ---------------------------------------------------------------------------
# Server-side compaction
# ---------------------------------------------------------------------------


class Compacting(CallsBlob):
    """A provider that compacts server-side, answering with a marker."""

    supports_compaction = True

    def __init__(self) -> None:
        super().__init__(calls=0)
        self.compacted: list[list[dict]] = []
        self.compact_tools: list[list[dict] | None] = []
        self.compact_models: list[str | None] = []
        self.fail = False

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.prompts.append([dict(m) for m in messages])
        return LLMResponse(content="done", finish_reason="stop", usage={"prompt_tokens": 90, "completion_tokens": 2})

    def replays_compaction(self, marker, model):
        return marker.get("provider") == "p" and marker.get("model") == model

    async def compact(self, messages, tools=None, model=None, reasoning_effort=None):
        if self.fail:
            raise RuntimeError("backend refused")
        self.compacted.append([dict(m) for m in messages])
        self.compact_tools.append(tools)
        self.compact_models.append(model)
        marker = build.marker({"provider": "p", "model": model, "items": [{"type": "opaque", "id": "summary"}]})
        return marker, {"prompt_tokens": 90, "completion_tokens": 300}


def _marker_of(prompt: list[dict]) -> dict | None:
    return next((m for m in prompt if "compaction" in m), None)


def _compacting_loop(tmp_path, window: int, ratio: float = 0.8):
    provider = Compacting()
    settings = AgentLoopSettings(model=MODEL, providers=_windows(model=window))
    settings.context.server_compact_ratio = ratio
    return provider, AgentLoop(provider, tmp_path, settings)


async def test_a_large_prompt_is_compacted_by_the_backend_after_the_turn(tmp_path) -> None:
    provider, loop = _compacting_loop(tmp_path, window=100)

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    assert len(provider.compacted) == 1, "90 of 100 tokens reaches the 0.8 threshold"
    assert provider.compacted[0][0]["role"] == "system", "the prompt as sent, system message included"
    stored = loop.sessions.get_or_create("cli:t").messages
    assert stored[-1]["compaction"]["model"] == MODEL and text_of(stored[-1]) == "[compacted]"
    assert "timestamp" in stored[-1]
    usage = loop.usage_tracker.session_usage("cli:t")
    assert usage.counts.calls == 2, "the compaction call is accounted to the session"
    await loop.close_mcp()


async def test_a_prompt_under_the_threshold_or_a_disabled_ratio_is_left_alone(tmp_path) -> None:
    provider, loop = _compacting_loop(tmp_path, window=200)
    await loop.run_turn(_request(), _emit, lambda: [], stream=False)
    assert provider.compacted == []
    await loop.close_mcp()

    provider, loop = _compacting_loop(tmp_path / "off", window=100, ratio=0)
    await loop.run_turn(_request(), _emit, lambda: [], stream=False)
    assert provider.compacted == []
    await loop.close_mcp()


async def test_a_failed_compaction_keeps_the_history_and_the_turn(tmp_path) -> None:
    provider, loop = _compacting_loop(tmp_path, window=100)
    provider.fail = True

    outcome = await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    assert outcome.text == "done"
    assert not any("compaction" in m for m in loop.sessions.get_or_create("cli:t").messages)
    await loop.close_mcp()


async def test_the_next_turn_sends_the_marker_and_not_the_history_it_stands_for(tmp_path) -> None:
    """Measured before this: every projection between the session log and the
    request dropped the ``compaction`` key, so the marker never reached the
    provider. The backend compacted the same conversation again every turn
    while the request kept replaying the history the summary had replaced."""
    provider, loop = _compacting_loop(tmp_path, window=100)

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)
    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    second = provider.prompts[1]
    marker = _marker_of(second)
    assert marker is not None and marker["compaction"]["items"] == [{"type": "opaque", "id": "summary"}]
    assert compaction_boundary(second, provider, MODEL) == second.index(marker)
    sent = wire_history(second, provider, MODEL)
    assert sent[0]["role"] == "system", "the instructions are not part of what a marker replaces"
    assert not any(text_of(m) == "done" for m in sent), "the first turn's reply is inside the summary"
    assert not any(text_of(m) == "done" for m in second), (
        "selection excludes what the marker stands for, so the request does not carry it either"
    )
    assert any(text_of(m) == "done" for m in loop.sessions.get_or_create("cli:t").messages), (
        "and the log still holds it, for a later turn on a model that replays nothing"
    )
    await loop.close_mcp()


async def test_the_marker_survives_a_restart_and_a_fork(tmp_path) -> None:
    """Disk and fork kept the marker already; what the next turn built from
    them is what dropped it."""
    provider, loop = _compacting_loop(tmp_path, window=100)
    await loop.run_turn(_request(), _emit, lambda: [], stream=False)
    fork = loop.sessions.fork("cli:t")
    await loop.close_mcp()

    restarted, fresh = _compacting_loop(tmp_path, window=100)  # a second process over the same workspace
    await fresh.run_turn(_request(), _emit, lambda: [], stream=False)
    assert _marker_of(restarted.prompts[0]) is not None, "loaded from disk by a process that never compacted"

    assert fork is not None
    chat_id = fork.key.partition(":")[2]
    forked = TurnRequest(
        origin=Origin.USER,
        source=Source(channel="cli", chat_id=chat_id, sender_id="user", chat_type=ChatType.DM),
        text="go",
        conversation=fork.key,
    )
    await fresh.run_turn(forked, _emit, lambda: [], stream=False)
    assert _marker_of(restarted.prompts[1]) is not None, "the fork carries the boundary too"
    await fresh.close_mcp()


async def test_a_compaction_cancelled_mid_flight_leaves_the_finished_turn_on_disk(tmp_path) -> None:
    """The answer has already been shown by the time compaction runs; a Ctrl-C
    during that second call used to take the whole turn with it, because the
    only disk write came after."""
    provider, loop = _compacting_loop(tmp_path, window=100)
    entered = asyncio.Event()

    async def never_answers(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    provider.compact = never_answers
    task = asyncio.create_task(loop.run_turn(_request(), _emit, lambda: [], stream=False))
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    reloaded = SessionManager(tmp_path).get_or_create("cli:t")
    assert [text_of(m) for m in reloaded.messages][-1] == "done"
    assert not any("compaction" in m for m in reloaded.messages), "an unfinished compaction records nothing"
    await loop.close_mcp()


class WithheldButRoutes(TokenStrategy):
    """Withholds the catalogue and routes the call, as the contract allows."""

    @property
    def name(self) -> str:
        return "withheld_routes"

    async def before_llm_call(self, messages, tools, model):
        return messages, [], "primary/big"


async def test_compaction_addresses_the_call_that_was_made_not_the_session_binding(tmp_path) -> None:
    """Measured before this: the request was rebuilt from ``self.model`` and the
    full tool registry, so a routed call's 90-of-100,000 usage was compared
    against the bound model's 100-token window and the compaction went to the
    wrong model carrying schemas that call had never sent."""
    provider = Compacting()
    settings = AgentLoopSettings(model=MODEL, providers=_windows(model=100, big=100))
    settings.context.server_compact_ratio = 0.8
    loop = AgentLoop(provider, tmp_path, settings)
    for n in range(5):
        loop.tools.register(Verbose(n))
    loop.strategies.register(WithheldButRoutes())

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    assert provider.compact_models == ["primary/big"], "the model that answered, not the one bound"
    assert provider.compact_tools == [[]], "the catalogue that call sent, not the registry"
    assert provider.compacted[0][:-1] == provider.prompts[0], "what the call sent"
    assert text_of(provider.compacted[0][-1]) == "done", "plus the answer the marker is filed behind"
    await loop.close_mcp()


async def test_a_resume_sends_the_marker_and_not_the_history_it_replaced(tmp_path) -> None:
    """Measured before this: a session resumed on a backend that compacts
    server-side was budgeted as if nothing would be replayed, and the selector
    deleted the marker under that budget. The request that followed carried
    300 kB of history the backend had already summarised.
    """
    provider = Compacting()

    written = SessionManager(tmp_path)
    session = written.get_or_create("cli:t")
    for builder in (build.user, build.assistant) * 3:
        session.record(builder("PRECOMPACT " + " word" * 4000))
    session.record(build.marker({"provider": "p", "model": MODEL, "items": [{"type": "opaque", "id": "summary"}]}))
    written.save(session)

    settings = AgentLoopSettings(model=MODEL, providers=_windows(model=(12_000, 1_000)))
    settings.context.server_compact_ratio = 0  # the request under test, not a fresh compaction
    loop = AgentLoop(provider, tmp_path, settings)

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    prompt = provider.prompts[0]
    assert _marker_of(prompt) is not None, "the boundary survived the budget that selected the history"
    sent = wire_history(prompt, provider, MODEL)
    assert not any("PRECOMPACT" in str(m.get("content")) for m in sent)
    stored = len(json.dumps(written.get_or_create("cli:t").messages))
    assert stored > 100_000, "the stored history really is the size the marker stands for"
    assert not any("PRECOMPACT" in str(m.get("content")) for m in prompt), (
        "and the request is the marker and what follows, not that"
    )
    await loop.close_mcp()


async def test_a_stale_marker_from_another_backend_keeps_the_whole_history(tmp_path) -> None:
    """The other half. A stale Codex marker left on a model id now served by an
    ordinary backend must not shrink that backend's budget: it will send the
    whole history, so the whole history has to be counted.
    """
    stale = build.marker({"provider": "openai-codex", "model": MODEL, "items": [{"type": "opaque"}]})
    history = [build.user("old " * 500), stale, build.user("next")]
    plain = CallsBlob()

    assert plain.replays_compaction(stale["compaction"], MODEL) is False, "nothing declared, so nothing replays"
    assert compaction_boundary(history, plain, MODEL) is None
    assert wire_history(history, plain, MODEL) == history, "and the prefix it stands for is still counted"


async def test_an_exhausted_turn_compacts_the_call_that_actually_ended_it(tmp_path) -> None:
    """Measured before this: the wrap-up call after the iteration budget ran
    out never updated the snapshot, so compaction summarised the request from
    before the last tool ran and filed its marker behind the result. The next
    turn on this model then had neither the result nor a summary covering it.
    """
    provider = ExhaustsThenWrapsUp()
    settings = AgentLoopSettings(model=MODEL, max_iterations=1, providers=_windows(model=(10_000, 1_000)))
    settings.context.server_compact_ratio = 0.8
    loop = AgentLoop(provider, tmp_path, settings)
    loop.tools.register(Blob(200))

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    stored = loop.sessions.get_or_create("cli:t").messages
    assert "compaction" in stored[-1]
    compacted = provider.compacted[0]
    assert [m["role"] for m in compacted] == ["system", "user", "assistant", "toolResult", "assistant"]
    result = next(m for m in stored if m["role"] == "toolResult")
    assert _tool_contents(compacted) == [text_of(result)], "every message the marker is filed behind was compacted"
    assert text_of(compacted[-1]) == "wrapped up", "including the wrap-up the turn ended on"
    await loop.close_mcp()


class ExhaustsThenWrapsUp(Compacting):
    """Spends the iteration budget on a tool, then answers the wrap-up call."""

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.prompts.append([dict(m) for m in messages])
        if len(self.prompts) == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="blob", arguments={})],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 1_000, "completion_tokens": 2},
            )
        return LLMResponse(
            content="wrapped up", finish_reason="stop", usage={"prompt_tokens": 9_500, "completion_tokens": 3}
        )


async def test_a_call_that_reported_no_prompt_usage_is_not_compacted(tmp_path) -> None:
    """A window nobody measured against a prompt nobody reported is not a
    reason to bill a full-prompt request."""
    provider, loop = _compacting_loop(tmp_path, window=100)

    async def silent(messages, tools=None, model=None, **kwargs):
        provider.prompts.append([dict(m) for m in messages])
        return LLMResponse(content="done", finish_reason="stop", usage={"prompt_tokens": 0, "completion_tokens": 0})

    provider.chat = silent

    await loop.run_turn(_request(), _emit, lambda: [], stream=False)

    assert provider.compacted == []
    await loop.close_mcp()


async def test_a_provider_without_server_compaction_is_never_asked(tmp_path) -> None:
    provider = Compacting()
    provider.supports_compaction = False
    loop = AgentLoop(provider, tmp_path, AgentLoopSettings(model=MODEL, providers=_windows(model=100)))
    await loop.run_turn(_request(), _emit, lambda: [], stream=False)
    assert provider.compacted == []
    await loop.close_mcp()


class TwoBilledCalls(LLMProvider):
    """A tool call and then the answer, each reporting its own usage."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def chat(self, messages, tools=None, model=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest(id="c1", name="blob", arguments={})],
                finish_reason="tool_calls",
                usage={"prompt_tokens": 100_000, "completion_tokens": 10_000, "total_tokens": 110_000},
            )
        return LLMResponse(
            content="done",
            finish_reason="stop",
            usage={"prompt_tokens": 200_000, "completion_tokens": 20_000, "total_tokens": 220_000},
        )

    def get_default_model(self) -> str:
        return MODEL


async def test_the_wrap_up_after_the_budget_runs_out_is_billed_like_any_other_call(tmp_path, monkeypatch) -> None:
    """A turn that ends at the iteration cliff still makes a second model call.

    It is the call the user actually reads, and it used to be the one call of a
    turn that reached no accounting at all: the wire reported $0.032 for a turn
    that spent $0.096, and left the first call's tokens and context beside the
    wrap-up's reply.
    """
    from opendde_harness.providers.rates import ListRates

    rates = ListRates(input=1e-5, output=1e-4, cache_read=0.0, cache_write=0.0)

    provider = TwoBilledCalls()
    # The provider states its own rates, the way the model service reports them
    # for the model it is about to call.
    provider.list_rates = lambda: rates
    # One iteration: the tool call exhausts the budget, so the answer comes
    # through the exhaustion wrap-up rather than through the loop.
    loop = _loop(tmp_path, provider, 100, max_iterations=1)

    outcome = await loop.run_turn(_request(), _emit, list, stream=False)

    first = rates.cost(100_000, 10_000, 0, 0)
    second = rates.cost(200_000, 20_000, 0, 0)

    assert provider.calls == 2
    assert outcome.usage_detail["list_cost_usd"] == pytest.approx(first + second)
    # And the wrap-up's own prompt is what the window figures describe: it is
    # the last call the turn made.
    assert outcome.usage_detail["context_used"] == 220_000
    assert outcome.usage_detail["prompt_tokens"] == 200_000


async def test_a_turn_reports_what_all_its_calls_cost_not_what_the_last_one_did(tmp_path, monkeypatch) -> None:
    """A tool-using turn makes several billed calls. The wire carries one
    figure, and it used to be the last call's: a turn that spent $0.096 across
    two calls told the status line $0.064."""
    from opendde_harness.providers.rates import ListRates

    # A price per call stated by the provider, so the assertion is about the
    # arithmetic and not about what any vendor charges this week.
    rates = ListRates(input=1e-5, output=1e-4, cache_read=0.0, cache_write=0.0)

    provider = TwoBilledCalls()
    provider.list_rates = lambda: rates
    loop = _loop(tmp_path, provider, 100)

    outcome = await loop.run_turn(_request(), _emit, list, stream=False)

    first = rates.cost(100_000, 10_000, 0, 0)
    second = rates.cost(200_000, 20_000, 0, 0)

    assert provider.calls == 2
    assert outcome.usage_detail["list_cost_usd"] == pytest.approx(first + second)
    assert outcome.usage_detail["list_cost_usd"] != pytest.approx(second)
    # The last call's own prompt still describes the window; only price is a
    # total of the turn.
    assert outcome.usage_detail["context_used"] == 220_000
