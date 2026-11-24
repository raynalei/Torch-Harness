# Original tools and skills, and the added model registry

This inventory uses the checkout's original `fdab4f0` baseline (before our
initial-round/model-tool changes). It describes this repository snapshot,
not every upstream release or a user's private plugins/MCP servers.

**Tools** are callable functions. **Skills** are instruction documents loaded on
demand; installing a skill does not install its models or make an API reachable.
Tool visibility depends on disabled plugins, blacklists, progressive disclosure,
optional dependencies and the selected workflow-agent role.

## Original main-assistant tools

| Name | Function |
| --- | --- |
| `read` | Read files |
| `write` | Write files |
| `edit` | Apply file edits |
| `ls` | List directories |
| `grep` | Search file content |
| `find` | Find files |
| `bash` | Execute permitted shell commands |
| `web_search` | Search the web using configured search support |
| `web_fetch` | Fetch web content |
| `spawn` | Start a configured subagent |
| `ask_user` | Ask structured user questions |
| `use_skill` | Load a selected skill body |
| `tool_search` | Discover tools when progressive disclosure is enabled |
| `tool_call` | Invoke a discovered tool in that mode |
| `understand_media` | Optional long-term-memory plugin tool for PDF/image/audio/etc. understanding |

See [registered model tools](protein-model-tools.md) and [Initial Round](initial-round.md)
for current design tools. See [local scientific analysis](protein-design.md#analyze-saved-results-locally)
for analysis of saved results.
