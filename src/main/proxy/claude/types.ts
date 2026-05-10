/**
 * Claude API Type Definitions
 * Based on Anthropic Messages API (https://docs.anthropic.com/en/api/messages)
 */

// ============================================================
// Claude Request Types
// ============================================================

export interface ClaudeMessageRequest {
  model: string
  messages: ClaudeMessage[]
  max_tokens: number
  metadata?: ClaudeMetadata
  stop_sequences?: string[]
  stream?: boolean
  system?: string | ClaudeSystemContentBlock[]
  temperature?: number
  top_p?: number
  top_k?: number
  tools?: ClaudeTool[]
  tool_choice?: ClaudeToolChoice
  thinking?: ClaudeThinkingConfig
}

export interface ClaudeThinkingConfig {
  type: 'enabled' | 'disabled'
  budget_tokens?: number
}

export interface ClaudeMetadata {
  user_id?: string
}

export interface ClaudeSystemContentBlock {
  type: 'text'
  text: string
  cache_control?: { type: 'ephemeral' }
}

export interface ClaudeMessage {
  role: 'user' | 'assistant'
  content: string | ClaudeContentBlock[]
}

export interface ClaudeContentBlock {
  type: 'text' | 'image' | 'tool_use' | 'tool_result' | 'thinking'
  text?: string
  thinking?: string
  signature?: string
  source?: ClaudeImageSource
  id?: string
  name?: string
  input?: Record<string, any>
  tool_use_id?: string
  content?: string | ClaudeContentBlock[]
  cache_control?: { type: 'ephemeral' }
}

export interface ClaudeImageSource {
  type: 'base64' | 'url'
  media_type: string
  data?: string
  url?: string
}

export interface ClaudeTool {
  name: string
  description?: string
  input_schema: {
    type: 'object'
    properties?: Record<string, any>
    required?: string[]
  }
  cache_control?: { type: 'ephemeral' }
}

export type ClaudeToolChoice =
  | 'auto'
  | 'any'
  | 'none'
  | { type: 'tool'; name: string }

// ============================================================
// Claude Response Types
// ============================================================

export interface ClaudeMessageResponse {
  id: string
  type: 'message'
  role: 'assistant'
  content: ClaudeContentBlock[]
  model: string
  stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null
  stop_sequence: string | null
  usage: ClaudeUsage
}

export interface ClaudeUsage {
  input_tokens: number
  output_tokens: number
  cache_creation_input_tokens?: number
  cache_read_input_tokens?: number
}

// ============================================================
// Claude Streaming Types
// ============================================================

export type ClaudeStreamEvent =
  | ClaudeMessageStartEvent
  | ClaudeMessageDeltaEvent
  | ClaudeMessageStopEvent
  | ClaudeContentBlockStartEvent
  | ClaudeContentBlockDeltaEvent
  | ClaudeContentBlockStopEvent
  | ClaudePingEvent
  | ClaudeErrorEvent

export interface ClaudeMessageStartEvent {
  type: 'message_start'
  message: ClaudeMessageResponse
}

export interface ClaudeMessageDeltaEvent {
  type: 'message_delta'
  delta: {
    stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null
    stop_sequence: string | null
  }
  usage: {
    output_tokens: number
  }
}

export interface ClaudeMessageStopEvent {
  type: 'message_stop'
}

export interface ClaudeContentBlockStartEvent {
  type: 'content_block_start'
  index: number
  content_block: ClaudeContentBlock
}

export interface ClaudeContentBlockDeltaEvent {
  type: 'content_block_delta'
  index: number
  delta: ClaudeDelta
}

export type ClaudeDelta =
  | ClaudeTextDelta
  | ClaudeToolUseDelta
  | ClaudeThinkingDelta
  | ClaudeSignatureDelta

export interface ClaudeTextDelta {
  type: 'text_delta'
  text: string
}

export interface ClaudeToolUseDelta {
  type: 'input_json_delta'
  partial_json: string
}

export interface ClaudeThinkingDelta {
  type: 'thinking_delta'
  thinking: string
}

export interface ClaudeSignatureDelta {
  type: 'signature_delta'
  signature: string
}

export interface ClaudeContentBlockStopEvent {
  type: 'content_block_stop'
  index: number
}

export interface ClaudePingEvent {
  type: 'ping'
}

export interface ClaudeErrorEvent {
  type: 'error'
  error: {
    type: string
    message: string
  }
}

// ============================================================
// Claude Error Response
// ============================================================

export interface ClaudeErrorResponse {
  type: 'error'
  error: {
    type: 'invalid_request_error' | 'authentication_error' | 'permission_error' | 'not_found_error' | 'request_too_large' | 'rate_limit_error' | 'api_error' | 'overloaded_error'
    message: string
  }
}
