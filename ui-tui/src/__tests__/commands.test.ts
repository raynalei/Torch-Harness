import { describe, expect, it } from 'vitest'

import { CommandRegistry, dispatch, looksLikeCommand, parseCommand } from '../commands/index.js'
import { rpcErrorFromFrame } from '../rpc/index.js'
import { createTestContext } from './commandContext.js'

const registry = new CommandRegistry()

function run(text: string, ...args: Parameters<typeof createTestContext>) {
  const harness = createTestContext(...args)

  return { harness, done: dispatch(registry, harness.context, text) }
}

describe('parsing', () => {
  it('recognises a command line', () => {
    expect(looksLikeCommand('/help')).toBe(true)
    expect(looksLikeCommand('/model gpt-5')).toBe(true)
    expect(looksLikeCommand('/')).toBe(true)
  })

  it('rejects prose and paths', () => {
    expect(looksLikeCommand('tell me about /etc')).toBe(false)
    expect(looksLikeCommand('/usr/bin/env')).toBe(false)
    expect(looksLikeCommand('')).toBe(false)
  })

  it('splits the name from the argument', () => {
    expect(parseCommand('/Model  gpt-5 --default ')).toEqual({
      arg: 'gpt-5 --default',
      name: 'model',
      text: '/Model  gpt-5 --default'
    })
    expect(parseCommand('/help')).toEqual({ arg: '', name: 'help', text: '/help' })
  })
})

describe('registry', () => {
  it('finds commands case-insensitively', () => {
    expect(registry.find('HELP')?.name).toBe('help')
    expect(registry.find('nope')).toBeUndefined()
  })

  it('carries exactly the set the owner settled on', () => {
    const names = registry.commands.map(command => command.name)

    // Twenty-five top-level and five namespaced actions. The list is the
    // whole point of the cut, so it is pinned exactly rather than checked for
    // membership: a command added back without a decision fails here.
    // `/login` and `/logout` are pi's own two, added with their pi names.
    expect(names).toEqual([
      'copy',
      'doctor',
      'export',
      'fast',
      'fork',
      'fullscreen',
      'help',
      'login',
      'logout',
      'mcp',
      'mcp:reload',
      'memory',
      'model',
      'new',
      'quiet-tools',
      'quit',
      'resume',
      'retry',
      'scoped-models',
      'sessions',
      'status',
      'task:follow',
      'task:logs',
      'tasks',
      'thinking',
      'title',
      'tracing',
      'undo',
      'verbose'
    ])
  })

  it('offers no administration and no session twins', () => {
    const names = new Set(registry.commands.map(command => command.name))

    // The CLI surface left the popup: provider, skill and compute settings,
    // and the six session commands the local ones already do. They all still
    // run when typed in full. `/tracing` came back: the dashboard is something
    // a user opens from the conversation, not administration.
    for (const gone of ['provider', 'skill', 'compute', 'plugins', 'compare', 'branch', 'task']) {
      expect(names.has(gone)).toBe(false)
    }
  })

  it('keeps a toggle that has a key out of the popup', () => {
    const names = new Set(registry.commands.map(command => command.name))

    // Shift+Tab, Ctrl+T and Ctrl+O do these; a command as well is a second
    // way to remember one thing.
    for (const gone of ['yolo', 'details', 'busy', 'token-usage', 'setup', 'history', 'context']) {
      expect(names.has(gone)).toBe(false)
    }
  })

  it('is sorted, so /help and the popup read alphabetically', () => {
    const names = registry.commands.map(command => command.name)

    expect(names).toEqual([...names].sort())
  })

  it('offers this registry and nothing from the gateway', () => {
    const { context } = createTestContext()
    const items = registry.slashCommands(context)

    // One source. The CLI surface used to be merged in here and brought 26
    // more rows, so the popup and the gateway could disagree; now there is
    // nothing to keep in step.
    expect(items.map(item => item.name)).toEqual(registry.commands.map(command => command.name))
    expect(items.some(item => item.name.startsWith('provider'))).toBe(false)
  })

  it('says what every command in the popup does', () => {
    const { context } = createTestContext()
    const items = registry.slashCommands(context)

    // A row with no description is a name and a shrug.
    expect(items.every(item => (item.description ?? '').trim().length > 0)).toBe(true)
  })

  it('exposes argument hints and completions to the popup', () => {
    const { context } = createTestContext()
    const thinking = registry.slashCommands(context).find(item => item.name === 'thinking')

    expect(thinking?.argumentHint).toBe('[level]')
    expect(thinking?.getArgumentCompletions?.('hi')).toEqual([{ label: 'high', value: 'high' }])
  })
})

describe('dispatch', () => {
  it('leaves a plain prompt alone', async () => {
    const { done } = run('write me a protein')

    expect(await done).toBe(false)
  })

  it('runs a local command', async () => {
    const { done, harness } = run('/quit')

    expect(await done).toBe(true)
    expect(harness.spies.quit).toHaveBeenCalledTimes(1)
    expect(harness.transport.methods).toEqual([])
  })

  it('sends an unknown command to slash.exec', async () => {
    const { done, harness } = run('/compare a b', { 'slash.exec': { output: 'compared' } })

    expect(await done).toBe(true)
    expect(harness.transport.calls[0]).toEqual({
      method: 'slash.exec',
      params: { command: 'compare a b', session_id: 'tui:abc' }
    })
    expect(harness.text()).toContain('compared')
  })

  it('reports a warning alongside the output', async () => {
    const { done, harness } = run('/compare', { 'slash.exec': { output: 'ok', warning: 'stale index' } })

    await done
    expect(harness.text()).toContain('warning: stale index')
  })

  it('reports a failing command rather than throwing', async () => {
    const { done, harness } = run('/compare', { 'slash.exec': new Error('gateway down') })

    expect(await done).toBe(true)
    expect(harness.text()).toContain('gateway down')
  })
})

describe('commands', () => {
  it('/help hands off to the panel the app builds, without touching focus', async () => {
    const { harness } = run(
      '/help',
      {},
      { catalog: { categories: [], commands: [], pairs: [['/compare', '/compare']], skill_count: 4 } }
    )

    await dispatch(registry, harness.context, '/help')

    expect(harness.spies.showHelp).toHaveBeenCalled()
    // Nothing was printed as a block: the panel is a transcript child, and the
    // registry/catalog/keybinding merge is tested in helpPanel.test.ts.
    expect(harness.text()).toBe('')
  })

  it('/status prints the gateway report', async () => {
    const { harness } = run('/status', { 'session.status': { output: 'Model: gpt-5' } })

    await dispatch(registry, harness.context, '/status')
    expect(harness.printed.some(line => line.title === 'Status' && line.text.includes('Model: gpt-5'))).toBe(true)
  })

  it('/status says so when there is no session', async () => {
    const harness = createTestContext({}, { sessionId: null })

    await dispatch(registry, harness.context, '/status')
    expect(harness.text()).toContain('no active session')
    expect(harness.transport.methods).toEqual([])
  })

  it('/title reads and writes', async () => {
    const harness = createTestContext({ 'session.title': { pending: false, session_key: 'tui:abc', title: 'Run 3' } })

    await dispatch(registry, harness.context, '/title')
    expect(harness.transport.calls[0]!.params).toEqual({ session_id: 'tui:abc' })
    expect(harness.text()).toContain('title: Run 3')

    await dispatch(registry, harness.context, '/title Run 3')
    expect(harness.transport.calls[1]!.params).toEqual({ session_id: 'tui:abc', title: 'Run 3' })
    expect(harness.info.title).toBe('Run 3')
  })

  it('/model opens the picker with no argument', async () => {
    const { harness } = run('/model')

    await dispatch(registry, harness.context, '/model')
    expect(harness.spies.openModelPicker).toHaveBeenCalledWith('session')

    await dispatch(registry, harness.context, '/model --default')
    expect(harness.spies.openModelPicker).toHaveBeenLastCalledWith('default')
  })

  it("/login opens pi's sign-in flow, with the provider it names", async () => {
    const { harness } = run('/login')

    await dispatch(registry, harness.context, '/login')
    expect(harness.spies.openLoginPicker).toHaveBeenCalledWith(undefined)

    await dispatch(registry, harness.context, '/login openai-codex')
    expect(harness.spies.openLoginPicker).toHaveBeenLastCalledWith('openai-codex')
  })

  it('/logout opens the sign-out list', async () => {
    const { harness } = run('/logout')

    await dispatch(registry, harness.context, '/logout')
    expect(harness.spies.openLogoutPicker).toHaveBeenCalled()
  })

  it('/model sends the qualified id and --default as the scope', async () => {
    const value = 'anthropic/claude-opus-5'
    const harness = createTestContext({ 'config.set': { applied: true, previous: null, value } })

    await dispatch(registry, harness.context, `/model ${value} --default`)

    // No provider param: the id names its provider, and a second word for it
    // used to win when the two disagreed.
    expect(harness.transport.calls[0]).toEqual({
      method: 'config.set',
      params: {
        key: 'model',
        scope: 'default',
        session_id: 'tui:abc',
        value
      }
    })
    expect(harness.info.model).toBe(value)
    expect(harness.text()).toContain(`Default model: ${value}`)
  })

  it('/model <text> that names no exact model opens the list with the text in its search', async () => {
    // pi's own rule for `/model <text>`.
    const harness = createTestContext({
      'config.set': rpcErrorFromFrame({ code: -32008, message: 'model_not_available' })
    })

    await dispatch(registry, harness.context, '/model opus')

    expect(harness.spies.openModelPicker).toHaveBeenCalledWith('session', 'opus')
    expect(harness.text()).not.toContain('error')
  })

  it("/scoped-models opens pi's scoped-models list", async () => {
    const harness = createTestContext()

    await dispatch(registry, harness.context, '/scoped-models')

    expect(harness.spies.openScopedModelsPicker).toHaveBeenCalled()
  })

  it('/model reports a switch the server did not apply', async () => {
    const harness = createTestContext({ 'config.set': { applied: false, previous: null, value: 'bogus' } })

    await dispatch(registry, harness.context, '/model bogus')

    expect(harness.text()).toContain('was not applied')
    expect(harness.info.model).toBeUndefined()
  })

  it('/thinking sets the overlay and remembers the effort', async () => {
    const harness = createTestContext({ 'model.overlay': { field: 'reasoning_effort', model: 'gpt-5', value: 'high' } })

    await dispatch(registry, harness.context, '/thinking high')

    expect(harness.transport.calls[0]!.params).toEqual({
      field: 'reasoning_effort',
      session_id: 'tui:abc',
      value: 'high'
    })
    expect(harness.info.effort).toBe('high')
  })

  it('/undo drops the exchange only when the server removed something', async () => {
    const harness = createTestContext({ 'session.undo': { removed: 0 } })

    await dispatch(registry, harness.context, '/undo')
    expect(harness.spies.dropLastExchange).not.toHaveBeenCalled()
    expect(harness.text()).toContain('nothing to undo')
  })

  it('/undo and /retry pass on how full the window is now', async () => {
    const undo = createTestContext({ 'session.undo': { context_used: 27_200, removed: 2 } })

    await dispatch(registry, undo.context, '/undo')

    // The last figure the footer had came from a call that saw the exchange
    // this removed, and only the gateway can say what is left.
    expect(undo.context.session.info().contextTokens).toBe(27_200)

    const retry = createTestContext(
      { 'session.undo': { context_used: 1_024, removed: 2 } },
      { retry: { onServer: true, text: 'fold it' } }
    )

    retry.transcript.push({ role: 'user', text: 'fold it' }, { role: 'assistant', text: 'done' })
    await dispatch(registry, retry.context, '/retry')

    expect(retry.context.session.info().contextTokens).toBe(1_024)
  })

  it('/undo and /retry pass on an unmeasured window as unmeasured', async () => {
    const undo = createTestContext({ 'session.undo': { context_used: null, removed: 2 } })

    await dispatch(registry, undo.context, '/undo')

    // Null is the gateway saying nothing measures this window: a compaction
    // marker its backend replays stands in front of the history. Dropping it
    // for not being a number left the footer on the removed exchange's
    // percentage, and disagreeing with what /status says about the session.
    expect(undo.context.session.info().contextTokens).toBeNull()

    const retry = createTestContext(
      { 'session.undo': { context_used: null, removed: 2 } },
      { retry: { onServer: true, text: 'fold it' } }
    )

    retry.transcript.push({ role: 'user', text: 'fold it' }, { role: 'assistant', text: 'done' })
    await dispatch(registry, retry.context, '/retry')

    expect(retry.context.session.info().contextTokens).toBeNull()
  })

  it('says nothing about the window when the server removed nothing', async () => {
    const harness = createTestContext({ 'session.undo': { removed: 0 } })

    await dispatch(registry, harness.context, '/undo')

    expect(harness.context.session.info().contextTokens).toBeUndefined()
  })

  it('/retry undoes, then sends the last prompt again', async () => {
    const harness = createTestContext(
      { 'session.undo': { removed: 2 } },
      { retry: { onServer: true, text: 'fold it' } }
    )

    harness.transcript.push({ role: 'user', text: 'fold it' }, { role: 'assistant', text: 'done' })

    await dispatch(registry, harness.context, '/retry')

    expect(harness.spies.dropLastExchange).toHaveBeenCalledTimes(1)
    expect(harness.spies.send).toHaveBeenCalledWith('fold it')
  })

  it('/retry after /undo resends the prompt the undo removed', async () => {
    const harness = createTestContext(
      { 'session.undo': { removed: 2 } },
      { retry: { onServer: true, text: 'prompt B' } }
    )

    harness.transcript.push(
      { role: 'user', text: 'prompt A' },
      { role: 'assistant', text: 'answer A' },
      { role: 'user', text: 'prompt B' },
      { role: 'assistant', text: 'answer B' }
    )

    await dispatch(registry, harness.context, '/undo')
    await dispatch(registry, harness.context, '/retry')

    // B's exchange is already gone, here and on the gateway: retrying it must
    // resend B and must not undo A's exchange on the way.
    expect(harness.spies.send).toHaveBeenCalledWith('prompt B')
    expect(harness.spies.dropLastExchange).toHaveBeenCalledTimes(1)
    expect(harness.transcript.map(entry => entry.text)).toEqual(['prompt A', 'answer A'])
  })

  it('/retry says so when nothing has been sent yet', async () => {
    const harness = createTestContext({ 'session.undo': { removed: 2 } })

    harness.transcript.push({ role: 'user', text: 'from a resumed transcript' })

    await dispatch(registry, harness.context, '/retry')

    expect(harness.spies.send).not.toHaveBeenCalled()
    expect(harness.text()).toContain('nothing to retry')
  })

  it('/sessions list prints the sessions', async () => {
    const harness = createTestContext({
      'session.list': {
        sessions: [{ id: 'tui:abc', message_count: 12, preview: 'p', started_at: 0, title: 'Antibody run' }]
      }
    })

    await dispatch(registry, harness.context, '/sessions')

    expect(harness.printed.find(line => line.title === 'Sessions')?.text).toContain('Antibody run')
  })

  it('/resume without an id opens the picker', async () => {
    const harness = createTestContext()

    await dispatch(registry, harness.context, '/resume')
    expect(harness.spies.pick).toHaveBeenCalledTimes(1)

    await dispatch(registry, harness.context, '/resume older')
    expect(harness.spies.resume).toHaveBeenCalledWith('tui:older')
  })

  it('/new refuses while a turn is running', async () => {
    const harness = createTestContext({}, { busy: true })

    await dispatch(registry, harness.context, '/new Fresh start')

    expect(harness.spies.create).not.toHaveBeenCalled()
  })

  it('/fork adopts the fork and leaves closing the parent to the adoption', async () => {
    const harness = createTestContext({
      'session.branch': { message_count: 3, session_id: 'tui:fork', title: 'Side quest' }
    })

    await dispatch(registry, harness.context, '/fork side')

    // One close, not two: the adoption leaves the old session, and the command
    // closing it as well sent `session.close` for the same id twice.
    expect(harness.transport.methods).toEqual(['session.branch'])
    expect(harness.spies.adopt).toHaveBeenCalledWith('tui:fork', 'Side quest')
    expect(harness.info.title).toBe('Side quest')
    expect(harness.text()).toContain('3 messages carried')
  })

  it('/fork waits for a running turn like any other switch', async () => {
    const harness = createTestContext(
      { 'session.branch': { message_count: 3, session_id: 'tui:fork', title: 'Side quest' } },
      { busy: true }
    )

    await dispatch(registry, harness.context, '/fork side')

    // Forking while a turn runs left the gateway running parent and child at
    // once, with the UI subscribed only to the child.
    expect(harness.transport.methods).toEqual([])
    expect(harness.spies.adopt).not.toHaveBeenCalled()
    expect(harness.text()).toContain('a turn is running')
  })

  it('/export reports where the file went', async () => {
    const harness = createTestContext({ 'session.export': { exported: true, path: '/tmp/run.md' } })

    await dispatch(registry, harness.context, '/export')
    expect(harness.text()).toContain('/tmp/run.md')
  })

  it('/export explains an ambiguous id', async () => {
    const harness = createTestContext({
      'session.export': { candidates: ['tui:a', 'tui:b'], exported: false, reason: 'ambiguous' }
    })

    await dispatch(registry, harness.context, '/export ab')
    expect(harness.text()).toContain('tui:a, tui:b')
  })

  it('/mcp:reload passes the confirmation through and reports the count', async () => {
    const harness = createTestContext({ 'reload.mcp': { ok: true, reloaded: 2, tools_changed: true } })

    await dispatch(registry, harness.context, '/mcp:reload always')

    expect(harness.transport.calls[0]!.params).toEqual({ always: true, confirm: true, session_id: 'tui:abc' })
    expect(harness.text()).toContain('reloaded 2 MCP server(s)')
    expect(harness.text()).toContain('tool list changed')
  })

  it('/doctor and /tracing go through slash.exec', async () => {
    const harness = createTestContext({ 'slash.exec': { output: 'all good' } })

    await dispatch(registry, harness.context, '/doctor --fix')
    await dispatch(registry, harness.context, '/tracing')
    await dispatch(registry, harness.context, '/tracing stop')

    expect(harness.transport.calls.map(call => (call.params as { command: string }).command)).toEqual([
      'doctor --fix',
      'tracing',
      'tracing stop'
    ])
  })

  it('says where a removed command went instead of failing', async () => {
    const harness = createTestContext({ 'slash.exec': { output: 'should not run' } })

    await dispatch(registry, harness.context, '/yolo')
    await dispatch(registry, harness.context, '/branch')
    await dispatch(registry, harness.context, '/reload-mcp')

    // Muscle memory outlives a release. None of these reached the gateway to
    // be told "unknown command" about something this app used to do.
    expect(harness.transport.calls).toEqual([])
    expect(harness.text()).toContain('/yolo is Shift+Tab')
    expect(harness.text()).toContain('/branch is now /fork')
    expect(harness.text()).toContain('/reload-mcp is now /mcp:reload')
  })

  it('treats a prototype property as an unknown command, not as guidance', async () => {
    const harness = createTestContext({ 'slash.exec': { output: 'unknown command: constructor' } })

    // `/constructor` and `/__proto__` find members of Object.prototype on any
    // plain-object table. The retired table used to be one, so the first sent
    // a function to a renderer that expects a string and threw outside the
    // error boundary, taking the app down instead of answering.
    await dispatch(registry, harness.context, '/constructor')
    await dispatch(registry, harness.context, '/__proto__')
    await dispatch(registry, harness.context, '/toString')

    expect(harness.transport.calls.map(call => (call.params as { command: string }).command)).toEqual([
      'constructor',
      '__proto__',
      'toString'
    ])
    expect(harness.text()).not.toContain('is not a function')
  })

  it('keeps a throw from an unknown name inside the error boundary', async () => {
    const harness = createTestContext({ 'slash.exec': new Error('gateway is gone') })

    await dispatch(registry, harness.context, '/constructor')

    // Answered in the transcript rather than rejected: nothing above dispatch
    // catches this, so a throw here reaches the app's own handler.
    expect(harness.text()).toContain('gateway is gone')
  })

  it('points a bare CLI group at the terminal, and still runs a real one', async () => {
    const harness = createTestContext({ 'slash.exec': { output: 'the provider table' } })

    await dispatch(registry, harness.context, '/provider')
    expect(harness.transport.calls).toEqual([])
    expect(harness.text()).toContain('ddeharness provider')

    // The group head is not a command; `provider list` is, and typing it in
    // full still works. It is the popup it left, not the app.
    await dispatch(registry, harness.context, '/provider list')

    expect(harness.transport.calls.map(call => (call.params as { command: string }).command)).toEqual(['provider list'])
  })

  it('/mcp lists the servers the session can reach', async () => {
    const harness = createTestContext({
      'session.info': {
        info: {
          mcp_servers: [
            { connected: true, name: 'filesystem', tool_count: 4, transport: 'stdio' },
            { connected: false, name: 'search', tool_count: 0, transport: 'streamableHttp' }
          ]
        }
      }
    })

    await dispatch(registry, harness.context, '/mcp')

    const printed = harness.text()

    expect(printed).toContain('filesystem')
    expect(printed).toContain('stdio')
    expect(printed).toContain('4 tools')
    // A server connects on the session's first turn, so before that "not
    // connected" is a stage rather than a fault.
    expect(printed).toContain('not connected')
  })

  it('/memory lists the instruction files, nearest last', async () => {
    const harness = createTestContext({
      'session.instructions': {
        changed: null,
        cwd: '/work/repo/api',
        error: null,
        files: [
          {
            display: '~/.opendde_harness/ODH.md',
            enabled: true,
            path: '/home/u/.opendde_harness/ODH.md',
            size: 900,
            skipped: false,
            truncated: false
          },
          {
            display: 'AGENTS.md',
            enabled: true,
            path: '/work/repo/AGENTS.md',
            size: 40000,
            skipped: false,
            truncated: true
          },
          {
            display: 'api/AGENTS.md',
            enabled: false,
            path: '/work/repo/api/AGENTS.md',
            size: 300,
            skipped: false,
            truncated: false
          }
        ]
      }
    })

    await dispatch(registry, harness.context, '/memory')

    const printed = harness.text()

    expect(harness.transport.calls[0]!.params).toEqual({ session_id: 'tui:abc' })
    expect(printed).toContain('~/.opendde_harness/ODH.md')
    expect(printed).toContain('900 B')
    // A file over the per-file cap is sent in part, and saying only its size on
    // disk would overstate what the model actually reads.
    expect(printed).toContain('first 32 KB only')
    expect(printed).toContain('off')
    // Order is the whole meaning of the list: the last file wins.
    expect(printed.indexOf('AGENTS.md')).toBeLessThan(printed.indexOf('api/AGENTS.md'))
  })

  it('/memory says which file has moved since the session started', async () => {
    const harness = createTestContext({
      'session.instructions': {
        changed: null,
        cwd: '/work/repo',
        error: null,
        files: [
          {
            changed: true,
            display: 'AGENTS.md',
            enabled: true,
            path: '/work/repo/AGENTS.md',
            size: 400,
            skipped: false,
            truncated: false
          }
        ]
      }
    })

    await dispatch(registry, harness.context, '/memory')

    // The agent can write files. Something it is following that is not what
    // this conversation started with is the line worth reading.
    expect(harness.text()).toContain('changed since this session started')
  })

  it('/memory is unmistakable when there is nothing to follow', async () => {
    const harness = createTestContext({
      'session.instructions': { changed: null, cwd: '/work/repo', error: null, files: [] }
    })

    await dispatch(registry, harness.context, '/memory')

    const printed = harness.text()

    // Not a quiet empty table: someone who expected their AGENTS.md to be read
    // has to learn that it was not found, and where to put one.
    expect(printed).toContain('No AGENTS.md or ODH.md')
    expect(printed).toContain('/work/repo')
    expect(printed).toContain('~/.opendde_harness/AGENTS.md')
  })

  it('/memory off passes the path through and reports what changed', async () => {
    const harness = createTestContext({
      'session.instructions': {
        changed: 'api/AGENTS.md',
        cwd: '/work/repo',
        error: null,
        files: [
          {
            display: 'api/AGENTS.md',
            enabled: false,
            path: '/work/repo/api/AGENTS.md',
            size: 300,
            skipped: false,
            truncated: false
          }
        ]
      }
    })

    await dispatch(registry, harness.context, '/memory off api/AGENTS.md')

    expect(harness.transport.calls[0]!.params).toEqual({
      action: 'off',
      path: 'api/AGENTS.md',
      session_id: 'tui:abc'
    })
    expect(harness.text()).toContain('api/AGENTS.md is off for this session')
  })

  it('/memory says why nothing was switched, rather than showing the same list again', async () => {
    const harness = createTestContext({
      'session.instructions': {
        changed: null,
        cwd: '/work/repo',
        error: 'no instruction file here is called NOTES.md',
        files: []
      }
    })

    await dispatch(registry, harness.context, '/memory off NOTES.md')
    expect(harness.text()).toContain('no instruction file here is called NOTES.md')

    // A verb it does not have never reaches the gateway: `/memory list` would
    // otherwise be sent as a path and come back as a file that does not exist.
    harness.transport.calls.length = 0
    await dispatch(registry, harness.context, '/memory list')
    expect(harness.transport.calls).toEqual([])
    expect(harness.text()).toContain('/memory takes on or off')
  })

  it('/mcp says so when nothing is configured', async () => {
    const harness = createTestContext({ 'session.info': { info: { mcp_servers: [] } } })

    await dispatch(registry, harness.context, '/mcp')

    expect(harness.text()).toContain('no MCP servers configured')
  })

  it('the task commands read the local root instead of the gateway', async () => {
    const harness = createTestContext({ 'slash.exec': { output: 'all good' } })

    await dispatch(registry, harness.context, '/tasks')
    await dispatch(registry, harness.context, '/tasks abc123')
  })

  it('/tasks hide takes the design bar down and /tasks show brings it back', async () => {
    const harness = createTestContext({ 'slash.exec': { output: 'all good' } })
    const visibility: boolean[] = []

    harness.context.tasks.setVisible = (visible: boolean) => {
      visibility.push(visible)
    }

    await dispatch(registry, harness.context, '/tasks hide')
    await dispatch(registry, harness.context, '/tasks show')

    expect(visibility).toEqual([false, true])
    expect(harness.text()).toContain('/tasks show brings it back')
  })

  it('/setup points at the CLI', async () => {
    const harness = createTestContext()

    await dispatch(registry, harness.context, '/setup')
    expect(harness.text()).toContain('ddeharness onboard')
  })

  it('/quiet-tools starts on and takes the four words the pi extension took', async () => {
    const harness = createTestContext()

    // The name and the words are the owner's habit from `pi-fold`, which is why
    // they are spelled exactly like it.
    expect(harness.quietTools).toBe(true)

    await dispatch(registry, harness.context, '/quiet-tools off')
    expect(harness.quietTools).toBe(false)
    expect(harness.text()).toContain('quiet tools: off')

    await dispatch(registry, harness.context, '/quiet-tools toggle')
    expect(harness.quietTools).toBe(true)

    await dispatch(registry, harness.context, '/quiet-tools')
    expect(harness.quietTools).toBe(false)

    await dispatch(registry, harness.context, '/quiet-tools on')
    expect(harness.quietTools).toBe(true)
    expect(harness.text()).toContain('ctrl+o expands one')
  })

  it('/quiet-tools status reports without changing anything', async () => {
    const harness = createTestContext()

    await dispatch(registry, harness.context, '/quiet-tools status')

    expect(harness.quietTools).toBe(true)
    expect(harness.text()).toContain('quiet tools: on')
    expect(harness.text()).toContain('hides its output')
  })

  it('/quiet-tools refuses a word it does not know', async () => {
    const harness = createTestContext()

    await dispatch(registry, harness.context, '/quiet-tools louder')

    expect(harness.quietTools).toBe(true)
    expect(harness.printed.at(-1)).toEqual({ text: 'usage: /quiet-tools [on|off|toggle|status]', tone: 'warn' })
  })

  it('/quiet-tools says the model is unaffected', async () => {
    const harness = createTestContext()

    await dispatch(registry, harness.context, '/quiet-tools on')

    // The one thing worth saying out loud: this is a renderer setting.
    expect(harness.text()).toContain('The model still gets every result in full.')
    expect(harness.transport.methods).toEqual([])
  })
})
