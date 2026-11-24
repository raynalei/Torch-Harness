// Turns and the queue: sending a prompt, cancelling one, draining what is
// waiting, and the row that says a turn is running.
//
// What lives here is everything whose lifetime is a turn rather than a
// session: the cancel in flight, the refusal count, the busy indicator and the
// window title that reflects both it and a waiting prompt.

import type { KeybindingsManager, TUI } from '@earendil-works/pi-tui'

import type { HarnessEditor } from '../components/editor.js'
import type { MessageQueue } from '../components/queue.js'
import type { StatusLine } from '../components/statusIndicator.js'
import type { SystemTone } from '../components/systemLine.js'
import type { Gateway } from '../gateway.js'
import type { GlobalKeyHandler } from '../lib/keybindings.js'
import type { Phase2 } from '../phase2.js'
import type { Phase3 } from '../phase3.js'
import type { Theme } from '../theme.js'
import type { TurnController, TurnStatus } from '../turn.js'

import { StatusIndicator } from '../components/statusIndicator.js'
import { BRAND } from '../content/brand.js'
import { pickVerb } from '../content/verbs.js'
import { keyText } from '../lib/keybindings.js'
import { setTerminalTitle } from '../lib/terminal.js'
import { TurnInProgressError } from '../rpc/index.js'
import { TurnNotStartedError } from '../turn.js'

/** How long to wait before re-sending a message the gateway refused because
 *  the previous turn was still unwinding, when no cancel is pending to wait
 *  for. The unwind is milliseconds; this is a floor, not an estimate. */
const REFUSED_RETRY_MS = 250

/** Give up re-sending after this many refusals in a row: the turn the gateway
 *  says is running is not one this UI knows about, and a queue that retries
 *  forever would hide that. */
const MAX_REFUSALS = 3

export interface TurnRunnerOptions {
  /** Why a session switch cannot happen right now, for the message Ctrl+K
   *  shows when the head cannot go out. */
  blockedReason: () => null | string
  editor: HarnessEditor
  gateway: Gateway
  keybindings: KeybindingsManager
  keys: () => GlobalKeyHandler
  /** The model the title names, or nothing before a session is open. */
  model: () => string | undefined
  now: () => number
  /** Built after this; reached at call time, as phase 3 is. */
  phase2: () => Phase2
  phase3: () => Phase3
  print: (text: string, tone?: SystemTone) => void
  queue: MessageQueue
  /** The session a turn error or a turn's end belongs to. */
  sessionKey: () => null | string
  /** The busy indicator's row, between the queued messages and the editor. */
  status: StatusLine
  theme: Theme
  tui: TUI
  turn: TurnController
}

export class TurnRunner {
  private indicator: null | StatusIndicator = null
  /** The cancel RPC in flight. The gateway answers it only once the cancelled
   *  turn has unwound, so nothing is sent until it resolves. */
  private pendingCancel: null | Promise<void> = null
  private refusals = 0
  private retryTimer: null | ReturnType<typeof setTimeout> = null
  /** A broker prompt is waiting for an answer. */
  private promptWaiting = false

  constructor(private readonly opts: TurnRunnerOptions) {}

  /** A cancel the gateway has not answered yet. It blocks sending the way a
   *  running turn does. */
  get cancelling(): boolean {
    return this.pendingCancel !== null
  }

  async startTurn(content: string): Promise<void> {
    const { print, queue, tui, turn } = this.opts

    try {
      await turn.send(content)
      this.refusals = 0
    } catch (err) {
      if (err instanceof TurnNotStartedError) {
        // The controller refused before anything left this process. Nothing
        // was spent, so the prompt goes back where it was and waits for the
        // drain rather than vanishing with the keystroke that sent it.
        queue.unshift(content)
        tui.requestRender()

        return
      }

      if (!(err instanceof TurnInProgressError)) {
        return
      }

      // The previous turn is still unwinding on the gateway. The message goes
      // back to the head of the queue, and out again once the cancel that is
      // pending resolves — or after a moment, when nothing is pending.
      queue.unshift(content)
      this.refusals += 1

      if (this.refusals >= MAX_REFUSALS) {
        this.refusals = 0
        print('the gateway still reports a turn in progress — press Ctrl+K to try again', 'warn')
        tui.requestRender()

        return
      }

      this.scheduleDrain(REFUSED_RETRY_MS)
    }
  }

  cancelTurn(): void {
    if (this.pendingCancel || !this.opts.turn.active) {
      return
    }

    // The gateway answers `turn.cancel` after the cancelled task has unwound,
    // which is later than the `error` event that ends the turn on screen. The
    // queue waits for the answer, not the event.
    this.pendingCancel = this.opts.turn.cancel().finally(() => {
      this.pendingCancel = null
      this.drain()
    })
  }

  scheduleDrain(delayMs: number): void {
    if (this.retryTimer) {
      clearTimeout(this.retryTimer)
    }

    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.drain()
    }, delayMs)
    this.retryTimer.unref?.()
  }

  /** Send the next queued message, unless something says not yet. `drain`
   *  reserves the head synchronously (the send marks the turn active before
   *  it awaits), so a prompt typed in the meantime queues behind it. */
  drain(): void {
    this.opts.queue.drain()
    this.opts.tui.requestRender()
  }

  /** Ctrl+K: send the head now if sending can happen at all. The queue is
   *  left exactly as it was when it cannot, so the prompt is still there when
   *  the turn ends and the drain picks it up. */
  submitQueueHead(): void {
    const result = this.opts.queue.submitHead()

    if (result === 'editing') {
      this.opts.print('that message is open in the editor — enter replaces it, esc leaves it as it was', 'warn')
      this.opts.tui.requestRender()

      return
    }

    if (result === 'blocked') {
      this.opts.print(this.opts.blockedReason() ?? 'not ready to send yet', 'warn')
      this.opts.tui.requestRender()
    }
  }

  /** Ctrl+C's idle rung: the draft *and* the queue behind it. Clearing only
   *  the editor left a queued message pending, so the next Ctrl+C landed on
   *  the same rung again and the UI could never be quit from there. */
  clearPendingInput(): void {
    this.opts.phase3().queueEditing.reset()
    this.opts.editor.setText('')
    this.opts.queue.clear()
    this.opts.tui.requestRender()
  }

  /**
   * Say that something which is not a turn is running, until it is not.
   *
   * A slash command that runs on the gateway is a round trip with nothing on
   * screen: a slow gateway command can take time and the UI
   * looked hung for all of them. The busy row already exists and already says
   * how to interrupt; this borrows it.
   *
   * A turn's own indicator outranks this. When one is up the row is already
   * saying the app is working, and replacing its verb with a command name
   * would lose the turn's elapsed clock.
   */
  showActivity(label: string): () => void {
    if (this.indicator) {
      return () => {}
    }

    // Same reasoning as a turn starting: this takes the row, and an offer made
    // before it is stale.
    this.opts.keys().disarmQuit()

    const indicator = new StatusIndicator(this.opts.tui, this.opts.theme, {
      message: `${label}…`,
      now: this.opts.now
    })

    this.indicator = indicator
    indicator.start()
    this.opts.status.set(indicator)
    this.opts.tui.requestRender()

    return () => {
      // Unless a turn started meanwhile and took the row over.
      if (this.indicator !== indicator) {
        return
      }

      indicator.dispose()
      this.indicator = null
      this.opts.status.set(undefined)
      this.opts.tui.requestRender()
    }
  }

  onTurnStatus(status: TurnStatus): void {
    if (status === 'running') {
      // A turn starting supersedes a standing offer to exit. The ladder takes
      // it back when it is next consulted, but a turn that starts and finishes
      // between two Ctrl+C presses is never consulted, and the second press
      // would take an offer made before any of this happened.
      this.opts.keys().disarmQuit()

      if (!this.indicator) {
        // The clock starts here and runs for the whole turn, retries included,
        // and so does the verb: one activity per turn, drawn when it starts.
        this.indicator = new StatusIndicator(this.opts.tui, this.opts.theme, {
          hint: () => this.interruptHint(),
          message: `${pickVerb()}…`,
          now: this.opts.now
        })
        this.indicator.start()
        this.opts.status.set(this.indicator)
      }
    } else {
      this.indicator?.dispose()
      this.indicator = null
      this.opts.status.set(undefined)
      this.opts.keys().disarm()
    }

    if (status === 'error') {
      // Whatever was waiting on this turn cannot be answered usefully now.
      this.opts.phase2().onTurnError(this.opts.sessionKey())
    } else if (status === 'idle') {
      // A cancelled turn ends idle, not error. Its question is just as dead,
      // and leaving it up would hold the queue until someone pressed Escape.
      this.opts.phase2().onTurnEnded(this.opts.sessionKey())
    }

    this.updateTitle(status)
  }

  /** `esc to interrupt`, from the live binding — the key is reconfigurable, and
   *  a hint that named the wrong one would be worse than none. `escape` is
   *  spelled the way every other terminal UI spells it on a status line. */
  private interruptHint(): string {
    const keys = keyText(this.opts.keybindings, 'app.interrupt').replace(/\bescape\b/g, 'esc')

    return keys ? `${keys} to interrupt` : ''
  }

  /** A prompt is waiting for an answer. It outranks the turn in the window
   *  title: the point of the cue is to be seen from another tab. */
  setPromptAttention(waiting: boolean): void {
    this.promptWaiting = waiting
    this.updateTitle(this.opts.turn.active ? 'running' : 'idle')
  }

  updateTitle(status: TurnStatus): void {
    const mark = this.promptWaiting ? '⚠' : status === 'running' ? '⏳' : '✓'

    setTerminalTitle(this.opts.tui.terminal, `${mark} ${this.opts.model() ?? BRAND.name}`)
  }

  /** The spinner's animation and the row's once-a-second clock have separate
   *  owners, and quitting mid-turn reaches neither through the turn: nothing
   *  reports the turn idle on the way out, so the row is cleared here. */
  dispose(): void {
    this.indicator?.dispose()
    this.indicator = null
    this.opts.status.dispose()

    if (this.retryTimer) {
      clearTimeout(this.retryTimer)
    }
  }
}
