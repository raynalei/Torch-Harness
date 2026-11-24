# Onboarding

The first `ddeharness` run starts the onboarding wizard in the terminal before the TUI opens when no provider is configured; without a terminal it exits and asks you to run `ddeharness onboard`. Run `ddeharness onboard` at any time to change settings. There is no wizard inside the TUI: `/setup` prints a reminder to exit and run the terminal command. Providers and models are the exception: `/model` connects a provider, stores its key and switches models from inside the UI (see the [TUI guide](tui.md)).

The three steps are **LLM**, **long-term memory** and **web search**. Memory runs on your default model through the model service, so its step asks one question (on or off) and needs no key, endpoint or model of its own; skip it with `--skip-memory`. The web-search step asks for a Brave Search API key (free tier: 2,000 queries a month, from https://api-dashboard.search.brave.com/app/keys); press Enter to skip, and `web_search` uses DuckDuckGo without a key. A provider with a hosted search of its own (the Codex login, OpenAI, Anthropic) searches through that regardless. Pass `--brave-api-key` to set it without prompts.

Onboarding does not launch a design task or install Docker/GPU drivers.

Long-term memory summarises each session into memories with the same model the
conversation runs on (a Codex sign-in included) and recalls them by keyword; there
is no embedding or rerank model. Design tasks still run with memory off, but do not
retain long-term cases and learned skills. Turn it on later with `ddeharness onboard`.

## Step 1: the LLM provider

The first step is the TUI's `/login`, at the terminal: pi's own sign-in, step
for step and word for word. **Select authentication method:** asks which way in
-- **Sign in with an account** or **Sign in with an API key** -- then **Select
provider to configure:** offers the providers that take that way in, and the
sign-in runs right there (pi's login-method menu, then the device code or the
sign-in link), or the key form asks for one masked key. The last option under
the key method, **OpenAI Compatible**, declares an endpoint of your own -- a
provider id, a base URL, an optional key and, only for an endpoint that
publishes no `GET /models`, the model ids -- exactly as the TUI's form does.
Both doors call the same gateway handlers, so what the wizard can connect is
what `/login` can, and a provider is written the same way from either.

The wizard offers the handful almost everybody picks -- OpenAI Codex, OpenAI,
Anthropic, Google, OpenRouter, DeepSeek, Moonshot AI, Z.AI, xAI -- and the
endpoint row; every other provider pi ships is a `/login` away inside the TUI
once the setup is done. Azure OpenAI is a tenant's own resource and is written
with `ddeharness provider set azure-openai-responses --base-url <resource>
--api azure-openai-responses --api-key <key>`; so is any endpoint on a wire
other than Chat Completions.

After the provider is connected the wizard does what `/login` leaves to
`/model`: it checks that the provider answers, asks which model is the
default, and sends one test message (`--skip-test` skips it). Without prompts,
`--provider` with `--api-key` connects one of pi's own and `--provider` with
`--base-url` (plus `--model` for an endpoint that publishes no list) declares
an endpoint; `--provider` alone goes straight to that provider, as `/login
<provider>` does.

## The providers section

The wizard writes `providers` in `~/.opendde_harness/config.json`, and the
section is [pi-ai](https://github.com/earendil-works/pi)'s own `models.json`
shape: a map from pi provider id to a provider declaration. So pi's own
documentation describes this section, and a key is written exactly once, with no
aliases and no second spelling.

```json
"agents": { "defaults": { "model": "openai-codex/gpt-5.5-codex" } },
"providers": {
  "openai-codex": { "login": "oauth" },
  "anthropic":    { "apiKey": "sk-ant-..." },
  "openrouter":   { "apiKey": "sk-or-...", "models": ["anthropic/claude-opus-4-5"] },
  "my-vllm": {
    "baseUrl": "http://gpu-box:8000/v1",
    "api": "openai-completions",
    "models": [
      { "id": "qwen3-32b", "name": "Qwen3 32B", "contextWindow": 131072, "maxTokens": 32768 }
    ]
  }
}
```

Four rules cover the whole section.

**A key is a pi provider id.** `anthropic`, `openai`, `openai-codex`, `google`,
`openrouter`, `deepseek`, `groq`, `xai`, `mistral`, `zai`, `minimax`,
`moonshotai` and the rest; `/login` in the TUI lists them all. Any other key is
a provider your config declares itself, and the name is yours to choose.

**`baseUrl` says which kind an entry is.** No address means one of pi's own: pi
carries the address, the protocol and the model catalogue, so the entry adds the
credential and nothing else. An address means you are declaring the provider,
and an address needs the protocol it serves, so `api` is required beside it --
one of `openai-completions`, `openai-responses`, `anthropic-messages`,
`azure-openai-responses`, `google-generative-ai`, `mistral-conversations`,
`openai-codex-responses`. It is declared and never guessed from the URL. The
same is true the other way round: `api` without `baseUrl` is refused, because
naming a protocol only means something for a provider you are declaring.

**A credential is a key, a sign-in, or the environment.** `apiKey` holds a key.
`"login": "oauth"` says the credential is a sign-in, which lives in the model
service's credential store and never in this file -- run
`ddeharness provider login <id>`. An entry with neither is still a declaration:
pi reads the vendor's own variable itself (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
and the rest), and a self-hosted server usually wants no key at all.

**A model id names its provider.** `agents.defaults.model` is
`"<provider>/<model>"` and the prefix is the only thing that says who serves it;
a bare id is refused rather than routed by its spelling. A provider you declare
must list its models, because pi has no catalogue for one. An entry in `models`
is either the id the endpoint serves or a row describing it -- `name`,
`contextWindow`, `maxTokens`, `reasoning`, `input`, `cost`, plus this project's
own `temperature`, `reasoningEffort` and `catalogModel`. Write a row for what no
catalogue can know: a deployment you sized yourself, a model newer than pi's
catalogue, or a deployment name (`catalogModel` then says which catalogue model
is behind it). `ddeharness provider model set <provider> <model>` writes one.

A config written by an earlier release is refused with one message naming the
change; nothing is migrated, and `ddeharness onboard` writes a fresh file.

## Model services

Configure independent model endpoints and workflow profiles using the
[deployment guide](protein-design.md).
