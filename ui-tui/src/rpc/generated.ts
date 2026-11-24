// AUTO-GENERATED — DO NOT EDIT — run `npm run gen:rpc`
//
// Source of truth: ui-tui/rpc-schema/openrpc.json (OpenRPC 1.2.6).
// Regenerate via: cd ui-tui && npm run gen:rpc
// Lint (drift check) via: cd ui-tui && npm run lint:rpc
//
// 86 method-scoped types (43 RPC methods × {Params, Result}) + all
// components/schemas + JSON-RPC 2.0 envelope types.

/* eslint-disable */
/* tslint:disable */

/**
 * Recursive JSON value type. Implemented as an unconstrained object in JSON Schema; downstream Pydantic uses typing.Any.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "JsonValue".
 */
export type JsonValue = string | number | boolean | null | unknown[] | {};
/**
 * Discriminated union of turn streaming events. The 'type' field is the discriminator.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnEvent".
 */
export type TurnEvent =
  | MessageStartEvent
  | EpisodeStartEvent
  | TurnRetryEvent
  | TurnUsageEvent
  | TokenDeltaEvent
  | ThinkingDeltaEvent
  | ToolStartEvent
  | ToolProgressEvent
  | ToolCompleteEvent
  | MessageCompleteEvent
  | TurnNoticeEvent
  | ErrorEvent
  | CronDeliveredEvent
  | CronMissedEvent
  | ProteinDesignProgressEvent;
/**
 * One step of pi's own sign-in flow, in pi's own shape and camelCase: four it reports (device_code, auth_url, info, progress) and two it waits for an answer to (select, manual_code). The 'type' field is the discriminator.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginStep".
 */
export type LoginStep =
  LoginDeviceCodeStep | LoginAuthUrlStep | LoginInfoStep | LoginProgressStep | LoginSelectStep | LoginManualCodeStep;
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ServerNotification".
 */
export type ServerNotification =
  | ApprovalRequestNotification
  | ApprovalClosedNotification
  | ClarifyRequestNotification
  | ConfirmRequestNotification
  | LoginStepNotification
  | TurnEventNotification;

/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionInfo".
 */
export interface SessionInfo {
  model: string;
  model_id: string;
  provider: string;
  reasoning_effort: string | null;
  context_window: number;
  lazy: boolean;
  skills: {
    [k: string]: string[];
  };
  tools: {
    [k: string]: string[];
  };
  usage: SessionUsage;
  version: string;
  /**
   * When this version shipped, from the changelog the package carries. Null when the install has no changelog to read.
   */
  release_date: string | null;
  cwd: string;
  /**
   * The AGENTS.md / ODH.md files this session sends, outermost first. Listed, never quoted.
   */
  project_instructions: ProjectInstructionFile[];
  /**
   * Configured MCP servers. They connect on the session's first turn, so a freshly created session reports them not yet connected.
   */
  mcp_servers: McpServerInfo[];
  /**
   * The model runs on a plan rather than per token, so no per-call price describes it. Read from the provider entry the model id names: a sign-in (`login: "oauth"`) is a plan, a key is metered.
   */
  subscription: boolean;
  /**
   * Something will compact this session before its context window runs out: the backend server-side, or the context engine here.
   */
  auto_compact: boolean;
  update_available?: string;
  update_command?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionUsage".
 */
export interface SessionUsage {
  input: number;
  output: number;
  cost_usd: number | null;
  /**
   * The session's spend at the vendor's published price, whoever is billing; what the footer opens on. Null when no model used has a published price.
   */
  list_cost_usd?: number | null;
  calls: number;
  context_max: number;
  context_source: string;
  /**
   * What the stored conversation holds, in tokens, as the server estimates it. Null when nothing measures it: a compaction marker this session's backend replays stands in front of the history, so what the next call sends is the marker and not the messages it replaced. A status line shows "?" rather than a percentage of superseded history.
   */
  context_used: number | null;
  context_percent: number | null;
}
/**
 * One AGENTS.md / ODH.md instruction file the session found. The list travels; the contents never do.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ProjectInstructionFile".
 */
export interface ProjectInstructionFile {
  /**
   * Absolute path, which is what /memory on|off names.
   */
  path: string;
  /**
   * How it is shown: relative to the work-tree root, ~/... for the user scope, absolute otherwise.
   */
  display: string;
  /**
   * Size on disk in bytes, which is not the size sent when truncated is true.
   */
  size: number;
  /**
   * Larger than the 32 KiB per-file cap, so only its head is sent.
   */
  truncated: boolean;
  /**
   * Found but not sent: the 128 KiB total was already spent by nearer files.
   */
  skipped: boolean;
  /**
   * False when /memory off switched it off for this session.
   */
  enabled: boolean;
  /**
   * Its content differs from what this session first saw in it. The agent can write files, so a change nobody at the keyboard made is worth showing.
   */
  changed: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "McpServerInfo".
 */
export interface McpServerInfo {
  name: string;
  transport: 'stdio' | 'sse' | 'streamableHttp';
  connected: boolean;
  tool_count: number;
}
/**
 * One row in the TUI session picker (gatewayTypes.ts SessionListItem).
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionListItem".
 */
export interface SessionListItem {
  /**
   * Full session_key: <channel>:<chat_id>.
   */
  id: string;
  message_count: number;
  preview: string;
  source: string;
  /**
   * Unix timestamp derived from created_at.
   */
  started_at: number;
  title: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionMessage".
 */
export interface SessionMessage {
  role: string;
  text?: string;
  context?: JsonValue;
  name?: JsonValue;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "McpToolInfo".
 */
export interface McpToolInfo {
  /**
   * Raw tool name (without mcp_<server>_ prefix).
   */
  name: string;
  description: string;
  /**
   * JSON Schema for the tool's input arguments.
   */
  parameters: {
    [k: string]: JsonValue;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SkillInfo".
 */
export interface SkillInfo {
  name: string;
  source: 'local' | 'remote';
  pinned: boolean;
  description: string;
  tags: string[];
}
/**
 * One provider row. `slug` is a pi provider id: the key of the `providers` entry it describes, and the prefix of every model id in `models`.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOptionProvider".
 */
export interface ModelOptionProvider {
  slug: string;
  name: string;
  authenticated: boolean;
  is_current: boolean;
  /**
   * How this provider is reached: a sign-in, an address this config declares, or a key.
   */
  auth_type: 'oauth' | 'endpoint' | 'key';
  /**
   * The environment variable that already supplies this provider's key, or null when none does.
   */
  key_env: string | null;
  /**
   * Qualified ids (`<provider>/<model>`), which is what config.set and model.add_model take.
   */
  models: string[];
  total_models: number;
  /**
   * False when only configured, curated and current models are included; request this slug to expand its catalogue.
   */
  models_loaded: boolean;
  /**
   * Every way in pi offers for this provider, in pi's own order, read from pi's own provider objects. Two of them is what a client offers a choice between; empty when the model service could not be asked.
   */
  auth_methods: ('oauth' | 'key')[];
  /**
   * pi's own label for the sign-in option (its `oauth.loginLabel`, e.g. "Sign in with SuperGrok or X Premium"), or null where pi carries none and its generic sentence applies.
   */
  login_label: string | null;
  /**
   * pi's own name for this provider's key (its `apiKey.name`, e.g. "Anthropic API key").
   */
  key_label: string | null;
  /**
   * A key must be typed before this provider can serve anything.
   */
  needs_api_key: boolean;
  /**
   * This provider is one the config declares, so its address and the wire it speaks must both be given.
   */
  needs_base_url: boolean;
  /**
   * The declared address as configured, for a form that prefills it; null for one of pi's own.
   */
  base_url: string | null;
  /**
   * The wire the declared address serves, as configured; null for one of pi's own, which carries its own.
   */
  api: string | null;
  warning: string;
  /**
   * Keyed by the model id as it appears in `models`.
   */
  model_labels: {
    [k: string]: ModelLabel;
  };
}
/**
 * How a model reads to a person: the name its declared row gives it, else the one the model service reports, else the id with its provider prefix dropped.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLabel".
 */
export interface ModelLabel {
  label: string;
  description?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "UsageSnapshot".
 */
export interface UsageSnapshot {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /**
   * Share of the last call's prompt served from the provider's cache, 0-100; null when the call sent no prompt or the provider stated no cache figure.
   */
  cache_hit_percent?: number | null;
  /**
   * Estimated cost of the last call in USD; null on a plan-billed provider, where no per-token figure exists.
   */
  cost_usd?: number | null;
  /**
   * What the call is worth at the vendor's published price, whoever is billing. What a plan-billed session can show beside its (sub) marker; null when no price is published for the model.
   */
  list_cost_usd?: number | null;
  context_used?: number;
  context_max?: number;
  context_percent?: number;
  /**
   * Which tier sized context_max; "unknown" with a zero context_max means no table lists the model.
   */
  context_source?: string;
  /**
   * The conversation was compacted after this call, so context_used describes a prompt that no longer exists. What the window holds is unknown until the next call reports; a status line shows "?" rather than the stale figure.
   */
  context_compacted?: boolean;
  /**
   * The session's total prompt tokens, cached ones included, from the same tracker /status reads. Every field above describes this turn or its last call; the session_* fields describe the session, so a footer showing the session's spend and a /status line reporting it cannot disagree.
   */
  session_input_tokens?: number | null;
  /**
   * The session's total output tokens.
   */
  session_output_tokens?: number | null;
  /**
   * What the vendor says the session has cost; null on a plan-billed provider, which states no per-token figure.
   */
  session_cost_usd?: number | null;
  /**
   * The session's spend at the vendor's published price, whoever is billing. This is the figure the footer shows. Null when no model used has a published price -- never zero for unknown.
   */
  session_list_cost_usd?: number | null;
  /**
   * How many billed model calls the session has made.
   */
  session_calls?: number | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CliResult".
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CliDispatchResult".
 */
export interface CliResult {
  /**
   * Rich-rendered output with ANSI SGR sequences.
   */
  stdout: string;
  /**
   * Error / warning output with ANSI SGR sequences.
   */
  stderr: string;
  /**
   * CLI command exit code; 0 = success.
   */
  exit_code: number;
  /**
   * Only present for timeout / not-dispatch-compatible cases (mirrors a JSON-RPC error code).
   */
  error_code?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CommandsCatalogResponse".
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CommandsCatalogResult".
 */
export interface CommandsCatalogResponse {
  /**
   * Every CLI command the gateway will run for a slash, with what it is and what it takes. The set is exactly what cli.dispatch accepts, so a bare group head (`provider`) is absent: it is not a command and dispatch refuses it.
   */
  commands: {
    /**
     * Space-separated argv without the leading slash, e.g. "provider list".
     */
    name: string;
    /**
     * One line from the command's own Typer help or docstring.
     */
    description: string;
    /**
     * Positional arguments, <required> and [optional], in declaration order, then any option the command cannot run without. Absent when the command takes none.
     */
    argument_hint?: string;
    /**
     * Seconds this command is allowed, where the ordinary slash timeout is not enough. Absent for everything else. The gateway applies the same figure itself; a client reads it to say that a command may take a while.
     */
    timeout_s?: number;
  }[];
  /**
   * RETIRED WITH THE INK FRONT-END. Group heads only, which is why an entry here can be a group rather than a command; `commands` is what a client should read. Kept because the Ink client gates on non-empty pairs (createGatewayEventHandler.ts) and degrades to direct slash.exec without them. Remove this field when that client is deleted at the cut-over.
   */
  pairs: [string, string][];
  /**
   * Ordered category list: '(top-level)' first then alphabetical group names. Fully filtered groups (e.g. tui) do not appear.
   */
  categories: string[];
  /**
   * Total skill count from skill_forge store SQL count; 0 if DB missing (warning field then populated).
   */
  skill_count: number;
  /**
   * Optional warning pushed to TUI activity strip (e.g. 'skill store not initialized').
   */
  warning?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "MessageStartEvent".
 */
export interface MessageStartEvent {
  type: 'message.start';
  payload: {
    turn_id: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "EpisodeStartEvent".
 */
export interface EpisodeStartEvent {
  type: 'episode.start';
  payload: {
    index: number;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnUsageEvent".
 */
export interface TurnUsageEvent {
  type: 'turn.usage';
  payload: {
    completion_tokens: number;
    reasoning_tokens: number;
    calls: number;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnRetryEvent".
 */
export interface TurnRetryEvent {
  type: 'turn.retry';
  payload: {
    attempt: number;
    total: number;
    reason: string;
    discard: boolean;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnNoticeEvent".
 */
export interface TurnNoticeEvent {
  type: 'turn.notice';
  payload: {
    kind: 'model_fallback' | 'delivery_failed';
    text: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TokenDeltaEvent".
 */
export interface TokenDeltaEvent {
  type: 'token.delta';
  payload: {
    text: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ThinkingDeltaEvent".
 */
export interface ThinkingDeltaEvent {
  type: 'thinking.delta';
  payload: {
    text: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ToolStartEvent".
 */
export interface ToolStartEvent {
  type: 'tool.start';
  payload: {
    tool_call_id: string;
    name: string;
    arguments: {
      [k: string]: JsonValue;
    };
    display?: string | null;
  };
}
/**
 * Legacy event shape retained for clients; no current Python producer.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ToolProgressEvent".
 */
export interface ToolProgressEvent {
  type: 'tool.progress';
  payload: {
    tool_call_id: string;
    preview: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ToolCompleteEvent".
 */
export interface ToolCompleteEvent {
  type: 'tool.complete';
  payload: {
    tool_call_id: string;
    /**
     * The head of what the tool returned (up to 1 KiB), for the transcript row; what ctrl+o expands to. Not what the model was given.
     */
    result_preview: string;
    /**
     * Whether the result outran the preview, so expanding the row shows its head only.
     */
    truncated: boolean;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "MessageCompleteEvent".
 */
export interface MessageCompleteEvent {
  type: 'message.complete';
  payload: {
    turn_id: string | null;
    usage: UsageSnapshot;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ErrorEvent".
 */
export interface ErrorEvent {
  type: 'error';
  payload: {
    code: number;
    message: string;
    reason?: 'cancelled_by_client' | 'internal';
    detail?: string;
  };
}
/**
 * Legacy event shape retained for clients; no current Python producer.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CronDeliveredEvent".
 */
export interface CronDeliveredEvent {
  type: 'cron.delivered';
  payload: {
    job_id: string;
    name: string;
    text: string;
    fired_at: string;
  };
}
/**
 * Legacy event shape retained for clients; no current Python producer.
 *
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CronMissedEvent".
 */
export interface CronMissedEvent {
  type: 'cron.missed';
  payload: {
    count: number;
    items: {
      name: string;
      scheduled_at: string;
      message: string;
    }[];
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ProteinDesignProgressEvent".
 */
export interface ProteinDesignProgressEvent {
  type: 'protein_design.progress';
  payload: {
    event_id: string;
    task_id: string;
    timestamp: string;
    event_type: 'task' | 'cycle' | 'phase' | 'agent' | 'skill' | 'tool' | 'fold' | 'gate' | 'memory';
    status: 'started' | 'progress' | 'completed' | 'failed';
    cycle: number | null;
    total_cycles: number | null;
    phase: string;
    actor: string;
    skill: string | null;
    tool: string | null;
    summary: string;
    duration_ms: number | null;
    candidate_count: number | null;
    error: string | null;
    has_details: boolean;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ApprovalRequestNotification".
 */
export interface ApprovalRequestNotification {
  jsonrpc: '2.0';
  method: 'approval.request';
  params: {
    approval_id: string;
    conversation_id: string;
    turn_id: string;
    tool_call_id: string;
    command: string;
    description: string;
    action_digest: string;
    created_at: number;
    expires_at: number;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ApprovalClosedNotification".
 */
export interface ApprovalClosedNotification {
  jsonrpc: '2.0';
  method: 'approval.closed';
  params: {
    approval_id: string;
    conversation_id: string;
    reason: 'allow' | 'deny' | 'cancelled' | 'timeout' | 'error';
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ClarifyRequestNotification".
 */
export interface ClarifyRequestNotification {
  jsonrpc: '2.0';
  method: 'clarify.request';
  params: {
    conversation_id: string;
    request_id: string;
    question: string;
    choices: string[];
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfirmRequestNotification".
 */
export interface ConfirmRequestNotification {
  jsonrpc: '2.0';
  method: 'confirm.request';
  params: {
    request_id: string;
    prompt: string;
    default: boolean;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginDeviceCodeStep".
 */
export interface LoginDeviceCodeStep {
  type: 'device_code';
  userCode: string;
  verificationUri: string;
  intervalSeconds?: number;
  expiresInSeconds?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginAuthUrlStep".
 */
export interface LoginAuthUrlStep {
  type: 'auth_url';
  url: string;
  instructions?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginInfoStep".
 */
export interface LoginInfoStep {
  type: 'info';
  message: string;
  links?: {
    url: string;
    label?: string;
  }[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginProgressStep".
 */
export interface LoginProgressStep {
  type: 'progress';
  message: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginSelectStep".
 */
export interface LoginSelectStep {
  type: 'select';
  message: string;
  options: {
    id: string;
    label: string;
    description?: string;
  }[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginManualCodeStep".
 */
export interface LoginManualCodeStep {
  type: 'manual_code';
  message: string;
  placeholder?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "LoginStepNotification".
 */
export interface LoginStepNotification {
  jsonrpc: '2.0';
  method: 'login.step';
  params: {
    login_id: string;
    provider: string;
    step: LoginStep;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnEventNotification".
 */
export interface TurnEventNotification {
  jsonrpc: '2.0';
  method: 'event';
  params: {
    subscription_id: string;
    event: TurnEvent;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionListParams".
 */
export interface SessionListParams {
  /**
   * Positive limits slice TUI sessions after sorting by updated_at descending; zero, negative and non-integer limits are ignored.
   */
  limit?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionListResult".
 */
export interface SessionListResult {
  sessions: SessionListItem[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCreateParams".
 */
export interface SessionCreateParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCreateResult".
 */
export interface SessionCreateResult {
  session_id: string;
  info: SessionInfo;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionResumeParams".
 */
export interface SessionResumeParams {
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionResumeResult".
 */
export interface SessionResumeResult {
  session_id: string;
  info: SessionInfo;
  messages: SessionMessage[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionDeleteParams".
 */
export interface SessionDeleteParams {
  /**
   * Full session_key as sent by the UI.
   */
  session_id?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionDeleteResult".
 */
export interface SessionDeleteResult {
  deleted: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionInfoParams".
 */
export interface SessionInfoParams {
  /**
   * Full session_key; absent answers for the defaults a new session would start on.
   */
  session_id?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionInfoResult".
 */
export interface SessionInfoResult {
  info: SessionInfo;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionMostRecentParams".
 */
export interface SessionMostRecentParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionMostRecentResult".
 */
export interface SessionMostRecentResult {
  session_id: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionTitleParams".
 */
export interface SessionTitleParams {
  /**
   * Full session_key.
   */
  session_id?: string;
  /**
   * When present, set as the new title; when absent, return the current title.
   */
  title?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionTitleResult".
 */
export interface SessionTitleResult {
  title: string | null;
  session_key: string;
  pending: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionClearParams".
 */
export interface SessionClearParams {
  /**
   * Full session_key to clear.
   */
  session_id?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionClearResult".
 */
export interface SessionClearResult {
  /**
   * The same session_key (no new id minted).
   */
  session_id: string;
  /**
   * True when the in-place wipe ran.
   */
  cleared: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionUndoParams".
 */
export interface SessionUndoParams {
  /**
   * Full session_key to undo.
   */
  session_id?: string;
  /**
   * Trailing turns to drop (role==user boundary).
   */
  n?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionUndoResult".
 */
export interface SessionUndoResult {
  /**
   * Messages dropped (0 = nothing to undo).
   */
  removed: number;
  /**
   * What the conversation holds now the exchange is gone, in tokens, so a status line can stop reporting a window that has not been that full since. Null where a compaction marker the backend replays stands in front of the history, which is the answer session.info gives for that same session. Absent when nothing was removed.
   */
  context_used?: number | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionExportParams".
 */
export interface SessionExportParams {
  /**
   * Session id, prefix or full key; omitted or unknown ids return exported=false, path=null, reason=not_found.
   */
  session_id?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionExportResult".
 */
export interface SessionExportResult {
  /**
   * True when a Markdown file was written.
   */
  exported: boolean;
  /**
   * Absolute path of the written file, or null on failure.
   */
  path: string | null;
  /**
   * Failure reason when not exported: not_found | ambiguous | write_failed.
   */
  reason?: string;
  /**
   * Candidate full keys when reason is ambiguous.
   */
  candidates?: string[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionInstructionsParams".
 */
export interface SessionInstructionsParams {
  /**
   * Omitted or list reports; on / off switch one file for this session only.
   */
  action?: 'list' | 'on' | 'off';
  /**
   * Which file to switch: its absolute path, its shown name, or its basename when unambiguous.
   */
  path?: string;
  /**
   * Whose view this is. on / off apply to this conversation alone; omitted describes the defaults a new session would start on.
   */
  session_id?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionInstructionsResult".
 */
export interface SessionInstructionsResult {
  /**
   * The directory the search ran from.
   */
  cwd: string;
  /**
   * Every file found, outermost first.
   */
  files: ProjectInstructionFile[];
  /**
   * The shown name of the file just switched, else null.
   */
  changed: string | null;
  /**
   * Why nothing was switched, in one line, else null.
   */
  error: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSendParams".
 */
export interface TurnSendParams {
  session_key: string;
  content: string;
  channel?: string | null;
  chat_id?: string | null;
  sender_id?: string | null;
  /**
   * @maxItems 64
   */
  media?: string[] | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSendResult".
 */
export interface TurnSendResult {
  turn_id: string;
  accepted: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSubscribeParams".
 */
export interface TurnSubscribeParams {
  session_key: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnSubscribeResult".
 */
export interface TurnSubscribeResult {
  subscription_id: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnUnsubscribeParams".
 */
export interface TurnUnsubscribeParams {
  subscription_id: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnUnsubscribeResult".
 */
export interface TurnUnsubscribeResult {
  unsubscribed: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnCancelParams".
 */
export interface TurnCancelParams {
  session_key: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TurnCancelResult".
 */
export interface TurnCancelResult {
  cancelled: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOptionsParams".
 */
export interface ModelOptionsParams {
  slug?: string | null;
  include_catalog?: boolean;
  refresh?: boolean;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOptionsResult".
 */
export interface ModelOptionsResult {
  model: string;
  provider: string;
  /**
   * agents.defaults.model: what new sessions start on.
   */
  default_model?: string;
  providers: ModelOptionProvider[];
  /**
   * By provider id, the endpoints a refresh could not ask, each with the sentence it failed with; `*` when the service itself could not be asked.
   */
  refresh_errors?: {
    [k: string]: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelSaveKeyParams".
 */
export interface ModelSaveKeyParams {
  slug: string;
  api_key?: string;
  base_url?: string | null;
  api?: string | null;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelSaveKeyResult".
 */
export interface ModelSaveKeyResult {
  provider: ModelOptionProvider;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelDeclareProviderParams".
 */
export interface ModelDeclareProviderParams {
  provider: string;
  base_url: string;
  model?: string;
  api_key?: string;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelDeclareProviderResult".
 */
export interface ModelDeclareProviderResult {
  provider: ModelOptionProvider;
  /**
   * How many models the endpoint published when asked just now.
   */
  discovered: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLogoutParams".
 */
export interface ModelLogoutParams {
  slug: string;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLogoutResult".
 */
export interface ModelLogoutResult {
  /**
   * False when there was nothing stored to forget.
   */
  forgotten: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelScopeParams".
 */
export interface ModelScopeParams {
  models?: string[] | null;
  write?: boolean;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelScopeResult".
 */
export interface ModelScopeResult {
  models: string[] | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOverlayParams".
 */
export interface ModelOverlayParams {
  field:
    | 'name'
    | 'description'
    | 'api'
    | 'catalog_model'
    | 'context_window'
    | 'max_tokens'
    | 'reasoning'
    | 'reasoning_effort'
    | 'temperature';
  value: string;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelOverlayResult".
 */
export interface ModelOverlayResult {
  model: string;
  field: string;
  value: string | number | boolean | null;
  /**
   * The window the loop runs with after the change, for a field that sizes a request.
   */
  context_window?: number | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLoginParams".
 */
export interface ModelLoginParams {
  provider: string;
  login_id?: string | null;
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLoginResult".
 */
export interface ModelLoginResult {
  /**
   * The id its login.step notifications carried, so a client can tell which flow finished.
   */
  login_id: string;
  provider: ModelOptionProvider;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLoginAnswerParams".
 */
export interface ModelLoginAnswerParams {
  login_id: string;
  answer: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLoginAnswerResult".
 */
export interface ModelLoginAnswerResult {
  /**
   * False when nothing was waiting: the login ended, or the browser callback answered first.
   */
  answered: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLoginCancelParams".
 */
export interface ModelLoginCancelParams {
  login_id: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ModelLoginCancelResult".
 */
export interface ModelLoginCancelResult {
  cancelled: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigGetParams".
 */
export interface ConfigGetParams {
  /**
   * If omitted, return all whitelisted fields. Unknown keys are silently dropped.
   */
  keys?: string[] | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigGetResult".
 */
export interface ConfigGetResult {
  config: {
    'tui.theme'?: JsonValue;
    'tui.show_token_usage'?: JsonValue;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigSetParams".
 */
export interface ConfigSetParams {
  key: string;
  value: JsonValue;
  session_id?: string;
  scope?: 'session' | 'default';
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfigSetResult".
 */
export interface ConfigSetResult {
  applied: boolean;
  previous: JsonValue;
  value?: string;
  scope?: 'session' | 'default';
  session_id?: string;
  applies_to_session?: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SystemHelloParams".
 */
export interface SystemHelloParams {
  client_version: string;
  client_capabilities?: string[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SystemHelloResult".
 */
export interface SystemHelloResult {
  server_version: string;
  server_capabilities: string[];
  session: {
    default_channel: 'tui';
    default_session_key: string;
  };
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SystemPingParams".
 */
export interface SystemPingParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SystemPingResult".
 */
export interface SystemPingResult {
  pong: true;
  server_time_ms: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SystemVersionParams".
 */
export interface SystemVersionParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SystemVersionResult".
 */
export interface SystemVersionResult {
  server_version: string;
  /**
   * OpenRPC info.version mirrored back to client.
   */
  schema_version: string;
  opendde_harness_version: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CliDispatchParams".
 */
export interface CliDispatchParams {
  /**
   * Pre-tokenized argv (TUI side has already shlex-split).
   */
  argv: string[];
  /**
   * Ink container width in cells; required for Rich Console wrapping.
   */
  width: number;
  /**
   * Override the default 30s timeout for long-running commands.
   */
  timeout_s?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SetupStatusParams".
 */
export interface SetupStatusParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SetupStatusResult".
 */
export interface SetupStatusResult {
  provider_configured: boolean;
  compute_configured?: boolean;
  error?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ReloadMcpParams".
 */
export interface ReloadMcpParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ReloadMcpResult".
 */
export interface ReloadMcpResult {
  ok: boolean;
  reloaded: number;
  tools_changed: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CommandsCatalogParams".
 */
export interface CommandsCatalogParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCloseParams".
 */
export interface SessionCloseParams {
  session_id?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionCloseResult".
 */
export interface SessionCloseResult {
  ok: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionBranchParams".
 */
export interface SessionBranchParams {
  session_id?: string;
  name?: string | null;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionBranchResult".
 */
export interface SessionBranchResult {
  session_id: string | null;
  title: string | null;
  message_count?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionStatusParams".
 */
export interface SessionStatusParams {
  session_id?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SessionStatusResult".
 */
export interface SessionStatusResult {
  output: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TerminalResizeParams".
 */
export interface TerminalResizeParams {
  cols?: number;
  rows?: number;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "TerminalResizeResult".
 */
export interface TerminalResizeResult {
  ok: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SlashExecParams".
 */
export interface SlashExecParams {
  command?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "SlashExecResult".
 */
export interface SlashExecResult {
  output: string;
  warning?: string;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CompleteSlashParams".
 */
export interface CompleteSlashParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CompleteSlashResult".
 */
export interface CompleteSlashResult {
  /**
   * @maxItems 0
   */
  items: never[];
  replace_from: 1;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CompletePathParams".
 */
export interface CompletePathParams {}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "CompletePathResult".
 */
export interface CompletePathResult {
  /**
   * @maxItems 0
   */
  items: never[];
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ApprovalRespondParams".
 */
export interface ApprovalRespondParams {
  approval_id?: string;
  session_id?: string;
  conversation_id?: string;
  choice?: 'allow' | 'deny';
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ApprovalRespondResult".
 */
export interface ApprovalRespondResult {
  ok: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfirmRespondParams".
 */
export interface ConfirmRespondParams {
  request_id?: string;
  answer?: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ConfirmRespondResult".
 */
export interface ConfirmRespondResult {
  ok: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ClarifyRespondParams".
 */
export interface ClarifyRespondParams {
  conversation_id?: string;
  request_id?: string;
  answer?: string;
  cancelled?: boolean;
}
/**
 * This interface was referenced by `OpenDDEHarnessRpcRoot`'s JSON-Schema
 * via the `definition` "ClarifyRespondResult".
 */
export interface ClarifyRespondResult {
  ok: boolean;
}

// ---- Schema-name aliases for structurally-deduplicated types ----
export type CliDispatchResult = CliResult;
export type CommandsCatalogResult = CommandsCatalogResponse;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 envelope (specs/tui-ipc.md §2.1/2.2/2.3/2.4)
// ---------------------------------------------------------------------------

export interface JsonRpcRequest<P = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  method: string;
  params: P;
}

export interface JsonRpcSuccess<R = unknown> {
  jsonrpc: '2.0';
  id: string | number;
  result: R;
}

export interface JsonRpcErrorObject {
  code: number;
  message: string;
  data?: unknown;
}

export interface JsonRpcErrorResponse {
  jsonrpc: '2.0';
  id: string | number;
  error: JsonRpcErrorObject;
}

export type JsonRpcResponse<R = unknown> = JsonRpcSuccess<R> | JsonRpcErrorResponse;

export interface JsonRpcNotification<P = unknown> {
  jsonrpc: '2.0';
  method: string;
  params: P;
}

export interface EventNotificationParams<E = unknown> {
  subscription_id: string;
  event: E;
}

export function isJsonRpcError<R>(
  resp: JsonRpcResponse<R>,
): resp is JsonRpcErrorResponse {
  return (resp as JsonRpcErrorResponse).error !== undefined;
}
