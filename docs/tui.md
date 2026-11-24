# Terminal UI

The terminal UI is where you describe a design, watch the agent work, and follow
the tasks it starts. Complete [installation](installation.md) and
[onboarding](onboarding.md) first.

## Launch

```bash
ddeharness tui
```

Bare `ddeharness` does the same. The UI opens on the alternate screen, which has
its own scrollback and search. `OPENDDE_HARNESS_TUI_FULLSCREEN=0` starts on the
plain terminal instead, and `/fullscreen on|off|status` switches at any time.

| Option | Purpose |
| --- | --- |
| `--check` | Start the child process and exit. It proves the bundle loads; it does not check a provider, model services, or the screen |
| `--dev` | Run the TypeScript source through `tsx` instead of the built bundle. Needs a source checkout with its frontend dependencies installed |
| `--color auto\|truecolor\|256\|16\|none` | Pin the color capability instead of detecting it |

## The screen

Top to bottom: the welcome panel, the transcript, the design-task bar, queued
messages, the status row and the editor, then the footer. The task bar, the
queue and the status row take no rows when they have nothing to say.

The welcome panel opens the session and stays at the top of the scrollback. It
says what this session is; where it runs is the footer's line, not a second copy
here:

| Row | Meaning |
| --- | --- |
| Brand lockup | The wordmark, dropped on a terminal too narrow or too short for it |
| `ϒ OpenDDE Harness v0.0.3 (2026-09-10)` | The installed version and the date it shipped, from the changelog the package carries |
| `model · provider · thinking · endpoint · session` | What this conversation runs on. The endpoint appears only for a provider that has more than one |
| `10 commands · 12 tools · 3 skills · 1 MCP server` | What is loaded. MCP servers are counted from the moment they are configured, and connect on the session's first turn |

The status row appears above the editor while a turn runs:

```
(◔_◔)  grafting CDRs… (12s · esc to interrupt)
```

The face animates, the verb is drawn once per turn, and the clock runs for the
whole turn including retries. It is how the product looks while it works, not a
setting.

The terminal title shows **⏳** while a turn runs, **⚠** while a prompt needs an
answer, and **✓** otherwise.

## The footer

Two lines. The first is where you are:

```
~/work/proj (main) • Antibody design
```

The working directory, the Git branch when there is one, and the session title
or id. When PyPI has a newer release the line opens with `↑ 0.0.4 · uv tool upgrade
opendde-harness` (the command for your kind of install) and the path is
shortened to make room for it; the same is said once in the transcript at
launch.

The second line is what the session has cost and what it runs on:

```
↑12k ↓~3.4k $0.031 (sub) 18.2%/200k (auto)          (openai) gpt-5 • high [fast]
```

| Group | Meaning |
| --- | --- |
| `↑12k` | Prompt tokens, summed over the session's calls |
| `↓3.4k` | Output tokens. A dim `~` in front means the tail of the figure is this UI's estimate of the reply still streaming; the next report from the model replaces it |
| `R…` `W…` `CH…%` | Cache reads, cache writes and hit rate. Shown only when the gateway reports cache counts |
| `$0.031` | What the tokens are worth at the vendor's published list price |
| `(sub)` | The model is billed by a plan, so that price is not what this conversation costs. The figure stays as the worth of the tokens |
| `18.2%/200k` | The last call's prompt against the model's context window. `?` instead of a percentage means a compaction has run and nothing has reported since; the window is still stated. The figure turns amber past 70% and red past 90% |
| `(auto)` | Something will compact this session before the window runs out, either the backend or the context engine here |
| `(openai) gpt-5 • high [fast]` | The model, its thinking level, and fast mode when it is on. The provider is named only when more than one is configured; a narrow terminal drops it first |

The counters are on unless `tui.show_token_usage` says otherwise; set it with
`ddeharness config set`. They cover this process only: switching sessions starts
them again, and a resumed transcript does not restore what it cost. `/status`
reports the gateway's own totals.

## Write and queue prompts

Enter submits. While a turn is running the prompt joins a queue and goes out when
the turn ends, in the order it was typed. Sending also waits for a cancellation
to finish and for any picker, prompt or sign-in to give the editor back.

Ctrl+K sends the first queued message immediately, which is the one case where
waiting for the turn is not what you want.

Alt+Up and Alt+Down step back through queued messages and load one into the
editor to rewrite; Enter puts it back where it was, Esc leaves it alone, and
Alt+X drops it. Ctrl+K sends the head of the queue as soon as sending is
possible. Two Enters on an empty editor within two seconds interrupt the running
turn.

Ctrl+G or Alt+G opens the draft in `$VISUAL` or `$EDITOR`; saving and closing
brings the text back, still unsent. Prompt history lives in
`~/.opendde_harness/.opendde_harness_history` and is shared with other sessions;
`OPENDDE_HARNESS_HOME` moves it.

## Commands

Type `/` in the editor to open the command popup, keep typing to filter, then Tab
or Enter to accept. Every command in the popup is one you reach for while talking
to the agent. Administration — providers, skills, the tracing dashboard
— lives in the CLI: run `ddeharness --help` in a terminal. Those commands still
work if you type one in full here, they are simply not offered.

A feature with more than one action namespaces them, `feature:action`, and the
bare name is the view: `/tasks` lists the design tasks, `/task:logs` reads one.

| Command | Purpose |
| --- | --- |
| `/help` | List the commands and hotkeys |
| `/quit` | Exit |
| `/status` | Live session facts and the gateway's usage totals |
| `/new [title]` | Start a fresh conversation |
| `/resume [id]` | Open the session picker, or resume an id |
| `/sessions` | List the saved sessions in the transcript |
| `/title [text]` | Show or set this session's title |
| `/fork [name]` | Fork this conversation and switch to the child |
| `/export [id]` | Write the session to a markdown file and print the path |
| `/undo` | Remove the last exchange, on the server and on screen |
| `/retry` | Send the last prompt again |
| `/copy [number]` | Copy a reply to the clipboard |
| `/model [provider/model] [--default]` | pi's model list: every model the connected providers serve. An exact id switches at once; any other text opens the list with it in the search. Enter moves this session only; `--default`, or ctrl+s on a model in the list, moves what new sessions start on (`agents.defaults.model`). |
| `/scoped-models` | pi's scoped models: choose which models `/model` shows under its **scoped** tab, and ctrl+s to save the choice (`agents.scopedModels`) |
| `/thinking [level]` | Open the level picker, or set the level for this model |
| `/login [provider]` | Configure provider authentication: pick the method, then the provider, then sign in. A provider named goes straight to it |
| `/logout` | Forget a stored credential: pick one of the sign-ins or keys this machine holds |
| `/fast [normal\|fast\|status]` | Switch the model between normal and fast service |
| `/verbose [cycle\|on\|off]` | How much tool output the agent reports |
| `/fullscreen [on\|off\|status]` | Alternate screen, or the plain terminal |
| `/doctor [--fix]` | Check provider and memory configuration |
| `/tracing [stop]` | Open the tracing dashboard (LLM, tool and memory spans); `stop` shuts it down |
| `/mcp` | The MCP servers this session can reach |
| `/mcp:reload [always]` | Reload them in the live session |
| `/tasks [id]` | Protein-design tasks: all of them, or one in detail |
| `/task:logs <id>` | Read one task's worker log |
| `/task:follow <id>` | Follow one task until it finishes |

Some commands that used to be here are gone, and typing one says where it went:
`/yolo` is Shift+Tab, `/details` is Ctrl+T and Ctrl+O, `/history` is the
transcript you are already looking at, and `/branch` is now `/fork`. The design
task bar is no longer switched on and off: it appears while a design is running
and draws nothing when none is.

Thinking levels are `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` and
`default`, which clears the per-model override. Which of them a model implements
is the model's business; the picker does not promise all seven.

## Keyboard

| Key | Action |
| --- | --- |
| Enter | Submit, or accept the highlighted completion |
| Esc | Close the completion popup, otherwise cancel the running turn |
| Ctrl+C | The ladder below |
| Ctrl+D | Quit on an empty editor, otherwise delete forward |
| Ctrl+T | Show or hide thinking across the transcript |
| Ctrl+O | Expand or collapse tool output across the transcript |
| Ctrl+K | Send the first queued message now |
| Alt+Up / Alt+Down | Edit an earlier or later queued message |
| Alt+X | Drop the queued message being edited |
| Ctrl+V | Paste from the terminal's clipboard, for a clipboard that is not on this host |
| Ctrl+G / Alt+G | Edit the prompt in `$VISUAL` or `$EDITOR` |
| Ctrl+L | Repaint the screen |
| Shift+Tab | Toggle yolo mode, or cycle the completion popup backward while it is open |

Ctrl+C is not an exit key on its own:

1. During a turn, it asks the gateway to cancel.
2. If the turn has still not ended, it resets this UI's own state and gives the
   prompt back. A queued message still waits for the cancellation to land.
3. Idle, it clears the draft and the queue.
4. Idle with neither, the first press offers to exit and says so in the row
   above the editor. A second press within two seconds takes the offer. The
   offer lapses by itself, on any other key, and when a turn starts, so the
   press that follows an ordinary cancel never exits.

Ctrl+D is the explicit exit key and goes in one press on an empty editor.

Inside a picker or a prompt, Esc and Ctrl+C do that view's back or deny action
instead. On the alternate screen, Page Up, Page Down and the search keys scroll
and search the transcript; they are silenced while a picker owns the keyboard.

## Pickers and prompts

A picker replaces the editor in place and gives the draft back when it closes.
Type to search, ↑/↓ to move, Enter to choose, Esc to go back.

`/model` is pi's model selector: one flat list of every model the connected
providers serve, each row `id [provider]`, the current model marked and the
default badged, "Model Name:" under the list. It paints the last known list at
once and refreshes the declared endpoints' catalogs behind it ("Model catalogs
refreshed."), asking an endpoint again at most once a day, the way pi paces its
own catalogs. With a scope saved by `/scoped-models`, the list opens on the
**scoped** tab and Tab flips to **all**; without one it says, as pi does, that
only configured providers are shown and `/login` adds providers. `/scoped-models`
toggles models in and out of the scope (Enter), all of them (ctrl+a / ctrl+x) or
a whole provider (ctrl+p), reorders them (alt+↑/↓), and saves with ctrl+s;
until saved the choice is session-only.

A provider the config declares -- a relay, a gateway, a server of your own -- is
treated the way pi treats OpenRouter: its address and key are typed, and the
models it serves are read from its `GET /models`, so nothing is typed by hand
unless the endpoint publishes no list.

`/login` connects a provider. It asks pi's own question when pi offers two ways
in:

```
Select authentication method for xAI:
 → Sign in with SuperGrok or X Premium
   Sign in with an API key
```

The first label is the provider's own, as pi writes it, and pi's generic
**Sign in with an account** stands in where pi has none. A provider with one way
in is not asked. **Sign in with an API key** asks for a masked key and, where it
needs one, a base URL; the provider list's last row, **OpenAI Compatible**,
declares a provider the gateway has never heard of: a provider id, a base URL and
an optional key, written the way `ddeharness provider set <id> --base-url ...
--api openai-completions` writes it, and the endpoint is asked for its models
the moment it is declared. Model ids are typed only for an endpoint that
publishes none.

Signing in happens right there: the gateway runs pi's own flow and the picker
shows each step as it arrives — the login-method menu (browser or device code),
then the sign-in URL, or the device code and where to type it. The picker waits
until the sign-in finishes and comes back to the provider list with the provider
connected. Esc cancels the sign-in. `ddeharness provider login <provider>` still
does the same thing from a terminal.

`/login` runs the same steps from the other end, the way pi's own does: the
authentication method first, then **Select provider to configure:** filtered to
that method, then the sign-in. `/login <provider>` goes straight to one.
`/logout` lists the credentials this machine holds -- sign-ins and stored keys,
not a key the environment supplies -- and forgets the chosen one. What the
provider declares (an address, a wire, its models) stays, as with pi's own.

`/resume` lists saved sessions with their title, id, message count and age,
newest first, and loads more on request. Ctrl+D deletes the selected one after a
confirmation. A session switch waits for a running turn, a pending cancellation
and the queue to clear.

Three prompts come from the agent rather than from a command, and each replaces
the editor while it waits:

| Prompt | How to answer |
| --- | --- |
| **Approval required** | Read the command. `1` allows it once, `2` denies it; Esc denies. The countdown is the gateway's own deadline and expiry denies. There is no "always allow" |
| **The agent needs an answer** | Arrows and Enter, or `1`–`9` for the first nine choices. **Other** opens a text field; with no choices the field opens at once |
| **Confirm** | `y`/`n` or arrows and Enter. Esc answers no. Gateway confirmations count down 30 seconds and take their default on expiry |

Answers are not chat messages: they go back to the agent that asked and never
reach the transcript or prompt history.

## Thinking and tool output

Thinking arrives collapsed to a label; Ctrl+T or a click expands it. A tool call
draws a panel with the call, a progress line while it runs, and the first ten
lines of the result; Ctrl+O or a click expands it. A call running longer than
eight seconds grows one quiet reassurance row with its elapsed time.

`(truncated)` on a result means the gateway sent a shortened version. Expanding
the panel cannot recover what was never sent. Ctrl+T and Ctrl+O set the default
for every panel, and reach one a click expanded on its own.

## Design tasks

Initial Round runs can continue in a detached worker after the TUI closes.
Run `/protein-design initial-round-list` to discover saved runs. Pass the returned
directory to `/protein-design initial-round-status --run-dir <directory>` to
inspect progress or to `/protein-design initial-round-resume --run-dir <directory>`
to resume. See [Initial Round](initial-round.md).

The task bar and `/tasks` still read historical design snapshots. Use
`/task:logs <id>` to inspect saved worker logs.

## Resuming

`/resume` restores a session's stored messages through the same components a live
turn uses. The gateway stores the role, the text and a tool name, not arguments,
timings or usage, so a resumed tool panel shows the call and its result without
an argument line or a duration. Nothing is invented to fill them.

The footer's counters start from zero on a resumed session, and the context
figure is what the transcript holds until the first call of the new process
reports its own.

## Settings

Two configuration keys belong to this UI, both under `tui` in
`~/.opendde_harness/config.json`. Both are edited outside the TUI: the counters
with `ddeharness config set`, the palette by hand.

| Key | Effect |
| --- | --- |
| `tui.show_token_usage` | `false` hides the footer counters |
| `tui.theme` | `dark`, `light`, or `default` to follow the terminal. Any other name falls back to `default` |

The environment is read at startup:

| Variable | Behavior |
| --- | --- |
| `OPENDDE_HARNESS_TUI_FULLSCREEN` | `0`, `false`, `no` or `off` starts on the plain terminal |
| `OPENDDE_HARNESS_TUI_RESUME` | Resume this session key, such as `tui:20260911_150358_f62f37` |
| `OPENDDE_HARNESS_TUI_QUERY` | Submit this prompt or command once the session is open |
| `OPENDDE_HARNESS_TUI_THEME` | `dark` or `light` pins the palette |
| `OPENDDE_HARNESS_TUI_LIGHT` | `1`/`true` forces light, `0`/`false` forces dark. Beats `_THEME` |
| `OPENDDE_HARNESS_TUI_BACKGROUND` | A three- or six-digit hex background hint for the startup palette |
| `OPENDDE_HARNESS_TUI_COLOR` | `auto`, `truecolor`, `256`, `16` or `none`; `--color` overrides it |
| `OPENDDE_HARNESS_TUI_DISABLE_MOUSE` | `1` leaves the mouse to the terminal, so selection and scrolling are its own |
| `OPENDDE_HARNESS_TUI_SCROLL_SPEED` | Rows per wheel notch on the alternate screen |
| `NO_COLOR` | Any value turns color off and beats every setting above |

The palette comes from the first of these that names one:
`OPENDDE_HARNESS_TUI_LIGHT`, `OPENDDE_HARNESS_TUI_THEME`, `tui.theme`, the
background hint, `COLORFGBG`, and finally dark. The first three fix it as well
as choose it, so the terminal's own background reply corrects only a palette
that was guessed.

That reply is waited for before the first paint: startup asks the terminal what
its background is and gives it 60 ms to answer, which is one round trip over a
local pty and a network hop over ssh. A terminal that answers is not waited on a
moment longer; one that does not answer costs that much and then gets the guess.
Nothing is drawn until the question is settled, so a light terminal never shows
the dark palette and there is no corrective repaint to see.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `opendde-tui: no TTY` | Launch from a real terminal. Piped input exits 0 without opening the screen |
| Exit code 2 | The bundle or the source checkout is missing, or the frontend was started without a gateway socket. Launch through `ddeharness tui` |
| Exit code 3 | Startup failed at the handshake or the session. Read the printed error, then run `ddeharness status` or `ddeharness doctor` outside the UI |
| The terminal is left on the alternate screen after a kill -9 | Run `reset`. A normal start resets the screen, autowrap, mouse, focus and paste modes itself, at boot as well as on exit |
| Colors are wrong or absent | Check `NO_COLOR` and `OPENDDE_HARNESS_TUI_COLOR`, then pin the palette with `OPENDDE_HARNESS_TUI_THEME`. `--color truecolor` overrides detection for one launch |
| The screen does not repaint after a resize | Ctrl+L repaints. A terminal that does not send SIGWINCH leaves the UI at its old size |
| `@` completion offers nothing | Install `fd` (or `fdfind`) and put it on PATH, then restart |
| The external editor does not open | Set `$VISUAL` or `$EDITOR`, for example `EDITOR='code --wait'`. Quoted arguments work; pipelines and shell expansion do not |
| A relay or provider error in the transcript | The message is the provider's own. `/status` names the model and endpoint in use, `/model` switches, and `ddeharness doctor --probe` tests the credential outside the UI |
| `/fast` or `/verbose` says the gateway will not store it | Expected: those keys are not in the gateway's writable set, so the change lasts as long as the process |
| A design task reads nothing and names a directory as writable by other users | macOS only. Reads there are anchored by the path rather than by a directory descriptor, so every directory from `/` down to the task root must be owned by you or by root and not writable by group or others, unless it carries the sticky bit as `/tmp` does. `chmod go-w` the directory the message names |

See [troubleshooting](troubleshooting.md) for provider, compute and installation
failures.

## End-to-end tests

`ui-tui/e2e` drives the real launcher, gateway, agent loop and frontend under a
pseudo-terminal, with only the model's replies scripted. It needs no terminal of
your own and no provider credentials.

```sh
PYTHONPATH=. python ui-tui/e2e/run.py --scenario tool --artifacts /tmp/tui-e2e
PYTHONPATH=. python ui-tui/e2e/run.py --scenario all
PYTHONPATH=. python -m pytest -m e2e -q -s
```

The `e2e` marker keeps these out of the normal test run; `run.py --scenario`
names one of them, and `boot_pty.py` beside it covers a boot that fails after
the screen has been taken. Each scenario pins `OPENDDE_HARNESS_TUI_FULLSCREEN`
rather than inheriting the default, so the scenarios that read the screen get
the plain terminal and the `fullscreen` scenario gets the alternate one.

The token estimator needs its `cl100k_base` vocabulary already cached. The
harness copies it from `$TIKTOKEN_CACHE_DIR`, `$DATA_GYM_CACHE_DIR` or
`/tmp/data-gym-cache` into each fixture and fails clearly when it is absent; it
never downloads. Artifacts, including the terminal recording and the screen at
exit, stay in the artifact directory on success as well as failure. Keep that
path short: a deep one makes the working directory longer than the terminal is
wide, and scenarios that match on it will not find it.
