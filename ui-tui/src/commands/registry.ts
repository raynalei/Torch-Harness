// The command registry: what `/` completes to, and what `dispatch` looks up.

import type { SlashCommand } from '@earendil-works/pi-tui'

import type { Command, CommandContext } from './types.js'

import { authCommands } from './auth.js'
import { coreCommands } from './core.js'
import { mcpCommands } from './mcp.js'
import { memoryCommands } from './memory.js'
import { sessionCommands } from './session.js'
import { taskCommands } from './tasks.js'
import { viewCommands } from './view.js'

export class CommandRegistry {
  /** Every command, alphabetical. */
  readonly commands: Command[]

  private readonly byName = new Map<string, Command>()

  constructor(extra: Command[] = []) {
    this.commands = [
      ...authCommands,
      ...coreCommands(),
      ...mcpCommands,
      ...memoryCommands,
      ...sessionCommands,
      ...taskCommands(),
      ...viewCommands(),
      ...extra
    ].sort((a, b) => a.name.localeCompare(b.name))

    for (const command of this.commands) {
      this.byName.set(command.name, command)
    }
  }

  find(name: string): Command | undefined {
    return this.byName.get(name.toLowerCase())
  }

  /**
   * The items pi's autocomplete shows after `/`, which is this registry and
   * nothing else.
   *
   * The gateway's CLI surface used to be merged in here, and it brought 26
   * more entries: provider, skill and compute administration, and twins of
   * half the session commands. None of them is something you reach for while
   * talking to the agent, and a popup you scroll is one you stop reading. They
   * all still run — `cli.dispatch` accepts them, and typing one in full works
   * — but they are the CLI's surface, not this one's.
   *
   * One source also means the popup and the gateway cannot drift: there is
   * nothing to keep in step.
   */
  slashCommands(ctx: CommandContext): SlashCommand[] {
    return this.commands
      .filter(command => !command.hidden)
      .map(command => ({
        name: command.name,
        ...(command.description ? { description: command.description } : {}),
        ...(command.argumentHint ? { argumentHint: command.argumentHint } : {}),
        ...(command.getArgumentCompletions
          ? { getArgumentCompletions: (prefix: string) => command.getArgumentCompletions!(prefix, ctx) }
          : {})
      }))
  }
}
