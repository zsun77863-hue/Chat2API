/**
 * Claude ↔ OpenAI Format Converter
 * Bidirectional conversion between Anthropic Claude API and OpenAI Chat Completions API formats
 */

import type {
  ClaudeMessageRequest,
  ClaudeMessage,
  ClaudeContentBlock,
  ClaudeTool,
  ClaudeToolChoice,
  ClaudeMessageResponse,
  ClaudeUsage,
  ClaudeStreamEvent,
  ClaudeTextDelta,
  ClaudeToolUseDelta,
} from './types'
import type {
  ChatCompletionRequest,
  ChatMessage,
  ChatMessageContent,
  ChatCompletionTool,
  ChatCompletionToolChoice,
  ChatCompletionResponse,
  ChatCompletionMessageToolCall,
  ToolCall,
} from '../types'

// ============================================================
// Claude → OpenAI Request Conversion
// ============================================================

/**
 * Convert a Claude Messages API request to an OpenAI Chat Completions API request
 */
export function claudeRequestToOpenAI(claudeReq: ClaudeMessageRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = []

  // Handle system prompt
  if (claudeReq.system) {
    const systemContent = typeof claudeReq.system === 'string'
      ? claudeReq.system
      : claudeReq.system.map(block => block.text).join('\n')
    messages.push({
      role: 'system',
      content: systemContent,
    })
  }

  // Convert messages
  for (const msg of claudeReq.messages) {
    const converted = convertClaudeMessageToOpenAI(msg)
    if (converted) {
      if (Array.isArray(converted)) {
        messages.push(...converted)
      } else {
        messages.push(converted)
      }
    }
  }

  const openaiReq: ChatCompletionRequest = {
    model: claudeReq.model,
    messages,
    stream: claudeReq.stream || false,
  }

  // Map parameters
  if (claudeReq.max_tokens !== undefined) {
    openaiReq.max_tokens = claudeReq.max_tokens
  }
  if (claudeReq.temperature !== undefined) {
    openaiReq.temperature = claudeReq.temperature
  }
  if (claudeReq.top_p !== undefined) {
    openaiReq.top_p = claudeReq.top_p
  }
  if (claudeReq.stop_sequences && claudeReq.stop_sequences.length > 0) {
    openaiReq.stop = claudeReq.stop_sequences.length === 1
      ? claudeReq.stop_sequences[0]
      : claudeReq.stop_sequences
  }
  if (claudeReq.metadata?.user_id) {
    openaiReq.user = claudeReq.metadata.user_id
  }

  // Convert tools
  if (claudeReq.tools && claudeReq.tools.length > 0) {
    openaiReq.tools = claudeReq.tools.map(convertClaudeToolToOpenAI)
    openaiReq.tool_choice = convertClaudeToolChoiceToOpenAI(claudeReq.tool_choice, claudeReq.tools)
  }

  return openaiReq
}

/**
 * Convert a single Claude message to OpenAI format
 */
function convertClaudeMessageToOpenAI(msg: ClaudeMessage): ChatMessage | ChatMessage[] | null {
  if (typeof msg.content === 'string') {
    return {
      role: msg.role,
      content: msg.content,
    }
  }

  // Content is an array of blocks
  const blocks = msg.content as ClaudeContentBlock[]

  // Check for tool_use blocks (assistant messages with tool calls)
  const toolUseBlocks = blocks.filter(b => b.type === 'tool_use')
  const textBlocks = blocks.filter(b => b.type === 'text')
  const toolResultBlocks = blocks.filter(b => b.type === 'tool_result')

  if (msg.role === 'assistant' && toolUseBlocks.length > 0) {
    // Assistant message with tool calls
    const textContent = textBlocks.map(b => b.text || '').join('')
    const toolCalls: ChatCompletionMessageToolCall[] = toolUseBlocks.map(block => ({
      id: block.id || `call_${Date.now().toString(36)}`,
      type: 'function' as const,
      function: {
        name: block.name || '',
        arguments: typeof block.input === 'object' ? JSON.stringify(block.input) : '{}',
      },
    }))

    return {
      role: 'assistant',
      content: textContent || null,
      tool_calls: toolCalls,
    }
  }

  if (msg.role === 'user' && toolResultBlocks.length > 0) {
    // User message with tool results - each tool result becomes a separate tool message
    const result: ChatMessage[] = []

    // Add any text content first
    const textContent = textBlocks.map(b => b.text || '').join('')
    if (textContent) {
      result.push({
        role: 'user',
        content: textContent,
      })
    }

    // Add tool results as separate tool messages
    for (const block of toolResultBlocks) {
      let resultContent: string
      if (typeof block.content === 'string') {
        resultContent = block.content
      } else if (Array.isArray(block.content)) {
        resultContent = (block.content as ClaudeContentBlock[])
          .filter(b => b.type === 'text')
          .map(b => b.text || '')
          .join('')
      } else {
        resultContent = ''
      }

      result.push({
        role: 'tool',
        content: resultContent,
        tool_call_id: block.tool_use_id || '',
      })
    }

    return result
  }

  // Regular text message with content array
  const textContent = textBlocks.map(b => b.text || '').join('')

  // Check for image blocks
  const imageBlocks = blocks.filter(b => b.type === 'image')
  if (imageBlocks.length > 0) {
    const content: ChatMessageContent[] = []
    if (textContent) {
      content.push({ type: 'text', text: textContent })
    }
    for (const imgBlock of imageBlocks) {
      if (imgBlock.source) {
        if (imgBlock.source.type === 'base64') {
          content.push({
            type: 'image_url',
            image_url: {
              url: `data:${imgBlock.source.media_type};base64,${imgBlock.source.data}`,
            },
          })
        } else if (imgBlock.source.type === 'url') {
          content.push({
            type: 'image_url',
            image_url: {
              url: imgBlock.source.url || '',
            },
          })
        }
      }
    }
    return {
      role: msg.role,
      content,
    }
  }

  return {
    role: msg.role,
    content: textContent || '',
  }
}

/**
 * Convert Claude tool definition to OpenAI format
 */
function convertClaudeToolToOpenAI(claudeTool: ClaudeTool): ChatCompletionTool {
  return {
    type: 'function',
    function: {
      name: claudeTool.name,
      description: claudeTool.description,
      parameters: claudeTool.input_schema as Record<string, any>,
    },
  }
}

/**
 * Convert Claude tool_choice to OpenAI format
 */
function convertClaudeToolChoiceToOpenAI(
  toolChoice: ClaudeToolChoice | undefined,
  tools: ClaudeTool[]
): ChatCompletionToolChoice | undefined {
  if (!toolChoice) return undefined

  if (toolChoice === 'auto') return 'auto'
  if (toolChoice === 'any') return 'required'
  if (toolChoice === 'none') return 'none'

  if (typeof toolChoice === 'object' && toolChoice.type === 'tool') {
    return {
      type: 'function',
      function: { name: toolChoice.name },
    }
  }

  return 'auto'
}

// ============================================================
// OpenAI → Claude Response Conversion
// ============================================================

/**
 * Convert an OpenAI Chat Completions response to a Claude Messages response
 */
export function openaiResponseToClaude(
  openaiResp: ChatCompletionResponse,
  requestId: string
): ClaudeMessageResponse {
  const choice = openaiResp.choices[0]
  if (!choice) {
    return createEmptyClaudeResponse(requestId, openaiResp.model)
  }

  const message = choice.message
  const content: ClaudeContentBlock[] = []

  // Add thinking content (reasoning_content from DeepSeek thinking models)
  if ((message as any)?.reasoning_content) {
    content.push({
      type: 'thinking',
      thinking: (message as any).reasoning_content,
    })
  }

  // Add text content
  if (message?.content) {
    content.push({
      type: 'text',
      text: message.content,
    })
  }

  // Add tool use blocks
  if (message?.tool_calls && message.tool_calls.length > 0) {
    for (const tc of message.tool_calls) {
      let input: Record<string, any> = {}
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {
        input = {}
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      })
    }
  }

  // If no content blocks, add empty text
  if (content.length === 0) {
    content.push({
      type: 'text',
      text: '',
    })
  }

  // Map stop reason
  const stopReason = mapFinishReasonToStopReason(choice.finish_reason)

  return {
    id: requestId,
    type: 'message',
    role: 'assistant',
    content,
    model: openaiResp.model,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: openaiResp.usage?.prompt_tokens || 0,
      output_tokens: openaiResp.usage?.completion_tokens || 0,
    },
  }
}

/**
 * Map OpenAI finish_reason to Claude stop_reason
 */
function mapFinishReasonToStopReason(
  finishReason: string | null | undefined
): 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null {
  if (!finishReason) return null
  switch (finishReason) {
    case 'stop':
      return 'end_turn'
    case 'length':
      return 'max_tokens'
    case 'tool_calls':
      return 'tool_use'
    case 'content_filter':
      return 'end_turn'
    default:
      return 'end_turn'
  }
}

/**
 * Create an empty Claude response
 */
function createEmptyClaudeResponse(id: string, model: string): ClaudeMessageResponse {
  return {
    id,
    type: 'message',
    role: 'assistant',
    content: [{ type: 'text', text: '' }],
    model,
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 },
  }
}

// ============================================================
// OpenAI Stream → Claude Stream Conversion
// ============================================================

/**
 * State tracker for converting OpenAI SSE stream to Claude SSE stream
 */
export class ClaudeStreamConverter {
  private requestId: string
  private model: string
  private messageStarted = false
  private currentContentBlockIndex = -1
  private currentContentBlockType: string | null = null
  private toolCallBuffers: Map<number, { id: string; name: string; arguments: string }> = new Map()
  private totalOutputTokens = 0
  private inputTokens = 0

  constructor(requestId: string, model: string) {
    this.requestId = requestId
    this.model = model
  }

  /**
   * Convert an OpenAI stream chunk to one or more Claude stream events
   */
  convertChunk(chunk: any): string[] {
    const events: string[] = []

    if (!chunk || !chunk.choices || chunk.choices.length === 0) {
      return events
    }

    const choice = chunk.choices[0]
    const delta = choice.delta

    // Start message on first chunk
    if (!this.messageStarted) {
      this.messageStarted = true
      // Extract usage from the chunk if available
      if (chunk.usage) {
        this.inputTokens = chunk.usage.prompt_tokens || 0
      }
      events.push(this.formatEvent({
        type: 'message_start',
        message: {
          id: this.requestId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: this.inputTokens,
            output_tokens: 0,
          },
        },
      }))
    }

    // Handle role delta (first chunk usually)
    if (delta?.role === 'assistant' && !delta.content && !delta.tool_calls && !delta.reasoning_content) {
      // Just the role announcement, no content yet
      return events
    }

    // Handle reasoning_content (thinking) - convert to Claude thinking blocks
    if (delta?.reasoning_content) {
      // Start a new thinking block if needed
      if (this.currentContentBlockType !== 'thinking') {
        // Close previous block if any
        if (this.currentContentBlockIndex >= 0) {
          events.push(this.formatEvent({
            type: 'content_block_stop',
            index: this.currentContentBlockIndex,
          }))
        }
        this.currentContentBlockIndex++
        this.currentContentBlockType = 'thinking'
        events.push(this.formatEvent({
          type: 'content_block_start',
          index: this.currentContentBlockIndex,
          content_block: { type: 'thinking', thinking: '' },
        }))
      }
      // Send thinking delta
      events.push(this.formatEvent({
        type: 'content_block_delta',
        index: this.currentContentBlockIndex,
        delta: { type: 'thinking_delta', thinking: delta.reasoning_content },
      }))
      this.totalOutputTokens++
    }

    // Handle text content
    if (delta?.content) {
      // Start a new text block if needed
      if (this.currentContentBlockType !== 'text') {
        // Close previous block if any
        if (this.currentContentBlockIndex >= 0) {
          events.push(this.formatEvent({
            type: 'content_block_stop',
            index: this.currentContentBlockIndex,
          }))
        }
        this.currentContentBlockIndex++
        this.currentContentBlockType = 'text'
        events.push(this.formatEvent({
          type: 'content_block_start',
          index: this.currentContentBlockIndex,
          content_block: { type: 'text', text: '' },
        }))
      }
      // Send text delta
      events.push(this.formatEvent({
        type: 'content_block_delta',
        index: this.currentContentBlockIndex,
        delta: { type: 'text_delta', text: delta.content },
      }))
      this.totalOutputTokens++
    }

    // Handle tool calls
    if (delta?.tool_calls && delta.tool_calls.length > 0) {
      for (const tc of delta.tool_calls) {
        const tcIndex = tc.index ?? 0

        // Initialize tool call buffer if new
        if (!this.toolCallBuffers.has(tcIndex)) {
          // Close previous content block if any
          if (this.currentContentBlockIndex >= 0 && this.currentContentBlockType !== 'tool_use') {
            events.push(this.formatEvent({
              type: 'content_block_stop',
              index: this.currentContentBlockIndex,
            }))
          }

          this.toolCallBuffers.set(tcIndex, {
            id: tc.id || `toolu_${Date.now().toString(36)}`,
            name: tc.function?.name || '',
            arguments: '',
          })

          // Start new tool_use block
          this.currentContentBlockIndex++
          this.currentContentBlockType = 'tool_use'

          events.push(this.formatEvent({
            type: 'content_block_start',
            index: this.currentContentBlockIndex,
            content_block: {
              type: 'tool_use',
              id: this.toolCallBuffers.get(tcIndex)!.id,
              name: tc.function?.name || '',
              input: {},
            },
          }))
        }

        // Append arguments
        const buffer = this.toolCallBuffers.get(tcIndex)!
        if (tc.function?.name) {
          buffer.name = tc.function.name
        }
        if (tc.function?.arguments) {
          buffer.arguments += tc.function.arguments
          // Send input_json_delta
          events.push(this.formatEvent({
            type: 'content_block_delta',
            index: this.currentContentBlockIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: tc.function.arguments,
            },
          }))
        }
      }
    }

    // Handle finish
    if (choice.finish_reason) {
      // Close current content block
      if (this.currentContentBlockIndex >= 0) {
        events.push(this.formatEvent({
          type: 'content_block_stop',
          index: this.currentContentBlockIndex,
        }))
      }

      // Map finish reason
      const stopReason = mapFinishReasonToStopReason(choice.finish_reason)

      // Send message delta and stop
      events.push(this.formatEvent({
        type: 'message_delta',
        delta: {
          stop_reason: stopReason,
          stop_sequence: null,
        },
        usage: {
          output_tokens: this.totalOutputTokens,
        },
      }))

      events.push(this.formatEvent({
        type: 'message_stop',
      }))
    }

    return events
  }

  /**
   * Generate the final events when stream ends without a finish_reason
   */
  finalize(): string[] {
    const events: string[] = []

    if (!this.messageStarted) {
      // No chunks were received, send empty response
      events.push(this.formatEvent({
        type: 'message_start',
        message: {
          id: this.requestId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }))
    }

    // Close current content block if still open
    if (this.currentContentBlockIndex >= 0 && this.currentContentBlockType !== null) {
      events.push(this.formatEvent({
        type: 'content_block_stop',
        index: this.currentContentBlockIndex,
      }))
    }

    // Send final message delta and stop
    events.push(this.formatEvent({
      type: 'message_delta',
      delta: {
        stop_reason: 'end_turn',
        stop_sequence: null,
      },
      usage: {
        output_tokens: this.totalOutputTokens,
      },
    }))

    events.push(this.formatEvent({
      type: 'message_stop',
    }))

    return events
  }

  /**
   * Format a Claude stream event as SSE data
   */
  private formatEvent(event: ClaudeStreamEvent): string {
    return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`
  }
}

// ============================================================
// Utility Functions
// ============================================================

/**
 * Generate a Claude-style message ID
 */
export function generateClaudeMessageId(): string {
  return `msg_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
}

/**
 * Convert OpenAI error response to Claude error format
 */
export function openaiErrorToClaude(error: any): { type: 'error'; error: { type: string; message: string } } {
  const errorType = error?.type || 'api_error'
  const errorMessage = error?.message || 'Unknown error'

  // Map OpenAI error types to Claude error types
  let claudeErrorType: string
  switch (errorType) {
    case 'invalid_request_error':
      claudeErrorType = 'invalid_request_error'
      break
    case 'authentication_error':
      claudeErrorType = 'authentication_error'
      break
    case 'rate_limit_error':
      claudeErrorType = 'rate_limit_error'
      break
    case 'insufficient_quota':
      claudeErrorType = 'permission_error'
      break
    default:
      claudeErrorType = 'api_error'
  }

  return {
    type: 'error',
    error: {
      type: claudeErrorType,
      message: errorMessage,
    },
  }
}
