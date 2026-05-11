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
      signature: generateThinkingSignature((message as any).reasoning_content),
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
 *
 * v1.8.1 fixes:
 * - SSE line buffering to handle chunk boundaries
 * - Empty string content handling (delta.content === "")
 * - Guaranteed message_start on first valid chunk
 * - Proper content_block_start before any content
 * - Correct [DONE] handling with finalize
 * - Proper stream end with message_stop
 */
/**
 * State tracker for converting OpenAI SSE stream to Claude SSE stream
 *
 * v1.8.2: Complete rewrite based on reference claude-code-proxy implementation
 * - message_start sent IMMEDIATELY before any content
 * - content_block_start for text sent IMMEDIATELY after message_start
 * - ping event sent after content_block_start
 * - Text deltas only for non-empty content
 * - Proper [DONE] handling with data: [DONE] after message_stop
 * - SSE line buffering for chunk boundaries
 * - cache_creation_input_tokens and cache_read_input_tokens in usage
 */
export class ClaudeStreamConverter {
  private requestId: string
  private model: string
  private messageStarted = false
  private textBlockStarted = false
  private textBlockClosed = false
  private messageStopped = false
  private toolCallBuffers: Map<number, { id: string; name: string; arguments: string }> = new Map()
  private currentToolBlockIndex = -1
  private totalOutputTokens = 0
  private inputTokens = 0
  private thinkingAccumulator = ''
  private thinkingBlockStarted = false
  private thinkingBlockClosed = false
  // SSE line buffer for handling chunk boundaries
  private lineBuffer = ''

  constructor(requestId: string, model: string) {
    this.requestId = requestId
    this.model = model
  }

  /**
   * Process a raw SSE chunk string (may contain partial lines)
   * Returns formatted Claude SSE events ready to write to the response
   */
  processSSEChunk(rawChunk: string): string {
    let output = ''
    this.lineBuffer += rawChunk
    const lines = this.lineBuffer.split('\n')
    // Keep the last (potentially incomplete) line in the buffer
    this.lineBuffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith('data:')) continue

      const raw = trimmed.slice(5).trim()

      if (raw === '[DONE]') {
        output += this.handleDone()
        continue
      }

      let parsed: any
      try {
        parsed = JSON.parse(raw)
      } catch {
        continue
      }

      output += this.convertChunk(parsed)
    }

    return output
  }

  /**
   * Flush any remaining buffered lines
   */
  flushRemaining(): string {
    let output = ''
    if (this.lineBuffer.trim()) {
      const trimmed = this.lineBuffer.trim()
      if (trimmed.startsWith('data:')) {
        const raw = trimmed.slice(5).trim()
        if (raw !== '[DONE]') {
          try {
            const parsed = JSON.parse(raw)
            output += this.convertChunk(parsed)
          } catch {
            // Skip unparseable
          }
        }
      }
      this.lineBuffer = ''
    }
    return output
  }

  /**
   * Send the initial message_start + content_block_start + ping events
   * This MUST be called before any content is sent
   */
  private ensureMessageStarted(): string {
    if (this.messageStarted) return ''
    this.messageStarted = true

    let output = ''

    // 1. message_start event
    output += this.formatEvent({
      type: 'message_start',
      message: {
        id: this.requestId,
        type: 'message',
        role: 'assistant',
        model: this.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: this.inputTokens,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: 0,
          output_tokens: 0,
        },
      },
    })

    // 2. content_block_start for text (index 0) - ALWAYS start with text block
    output += this.formatEvent({
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    })
    this.textBlockStarted = true

    // 3. ping event
    output += this.formatEvent({
      type: 'ping',
    })

    return output
  }

  /**
   * Convert an OpenAI stream chunk to Claude SSE events string
   */
  private convertChunk(chunk: any): string {
    let output = ''

    if (!chunk || !chunk.choices || chunk.choices.length === 0) {
      return output
    }

    const choice = chunk.choices[0]
    const delta = choice.delta

    // Extract usage from the chunk if available
    if (chunk.usage) {
      this.inputTokens = chunk.usage.prompt_tokens || 0
    }

    // Ensure message_start has been sent
    output += this.ensureMessageStarted()

    // Handle reasoning_content (thinking) - convert to Claude thinking blocks
    if (delta?.reasoning_content != null && delta.reasoning_content !== '') {
      // Close text block first if it was started and not yet closed
      if (this.textBlockStarted && !this.textBlockClosed) {
        output += this.formatEvent({
          type: 'content_block_stop',
          index: 0,
        })
        this.textBlockClosed = true
      }

      // Start thinking block if not started
      if (!this.thinkingBlockStarted) {
        this.thinkingBlockStarted = true
        output += this.formatEvent({
          type: 'content_block_start',
          index: 1,
          content_block: { type: 'thinking', thinking: '' },
        })
      }

      // Send thinking delta
      output += this.formatEvent({
        type: 'content_block_delta',
        index: 1,
        delta: { type: 'thinking_delta', thinking: delta.reasoning_content },
      })
      this.thinkingAccumulator += delta.reasoning_content
      this.totalOutputTokens++
    }

    // Handle text content - only send non-empty content
    if (delta?.content != null && delta.content !== '') {
      // If we were in a thinking block, close it and reopen text block
      if (this.thinkingBlockStarted && !this.thinkingBlockClosed) {
        // Send signature_delta before closing thinking block
        if (this.thinkingAccumulator) {
          const signature = generateThinkingSignature(this.thinkingAccumulator)
          output += this.formatEvent({
            type: 'content_block_delta',
            index: 1,
            delta: { type: 'signature_delta', signature },
          })
        }
        output += this.formatEvent({
          type: 'content_block_stop',
          index: 1,
        })
        this.thinkingBlockClosed = true
      }

      // If text block was closed (because thinking started), reopen it
      if (this.textBlockClosed) {
        const textIndex = this.thinkingBlockStarted ? 2 : 0
        output += this.formatEvent({
          type: 'content_block_start',
          index: textIndex,
          content_block: { type: 'text', text: '' },
        })
        this.textBlockClosed = false
      }

      // Send text delta
      const textIndex = this.thinkingBlockStarted && !this.thinkingBlockClosed ? 2 : 0
      output += this.formatEvent({
        type: 'content_block_delta',
        index: textIndex,
        delta: { type: 'text_delta', text: delta.content },
      })
      this.totalOutputTokens++
    }

    // Handle tool calls
    if (delta?.tool_calls && delta.tool_calls.length > 0) {
      for (const tc of delta.tool_calls) {
        const tcIndex = tc.index ?? 0

        // Initialize tool call buffer if new
        if (!this.toolCallBuffers.has(tcIndex)) {
          // Close text block if still open
          if (this.textBlockStarted && !this.textBlockClosed) {
            output += this.formatEvent({
              type: 'content_block_stop',
              index: 0,
            })
            this.textBlockClosed = true
          }

          // Close thinking block if still open
          if (this.thinkingBlockStarted && !this.thinkingBlockClosed) {
            if (this.thinkingAccumulator) {
              const signature = generateThinkingSignature(this.thinkingAccumulator)
              output += this.formatEvent({
                type: 'content_block_delta',
                index: 1,
                delta: { type: 'signature_delta', signature },
              })
            }
            output += this.formatEvent({
              type: 'content_block_stop',
              index: 1,
            })
            this.thinkingBlockClosed = true
          }

          this.toolCallBuffers.set(tcIndex, {
            id: tc.id || `toolu_${Date.now().toString(36)}`,
            name: '',
            arguments: '',
          })

          this.currentToolBlockIndex++
          const blockIndex = this.getNextBlockIndex()

          output += this.formatEvent({
            type: 'content_block_start',
            index: blockIndex,
            content_block: {
              type: 'tool_use',
              id: this.toolCallBuffers.get(tcIndex)!.id,
              name: tc.function?.name || '',
              input: {},
            },
          })
        }

        const buffer = this.toolCallBuffers.get(tcIndex)!

        if (tc.function?.name) {
          buffer.name = tc.function.name
        }
        if (tc.function?.arguments) {
          buffer.arguments += tc.function.arguments
          const blockIndex = this.getNextBlockIndex()
          output += this.formatEvent({
            type: 'content_block_delta',
            index: blockIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: tc.function.arguments,
            },
          })
        }
      }
    }

    // Handle finish_reason
    if (choice.finish_reason) {
      output += this.handleFinish(choice.finish_reason)
    }

    return output
  }

  /**
   * Get the next content block index based on current state
   */
  private getNextBlockIndex(): number {
    let index = 0
    if (this.textBlockStarted) index++
    if (this.thinkingBlockStarted) index++
    index += this.currentToolBlockIndex
    return index
  }

  /**
   * Handle finish_reason - close all open blocks and send message_stop
   */
  private handleFinish(finishReason: string): string {
    if (this.messageStopped) return ''
    let output = ''

    // Close text block if still open
    if (this.textBlockStarted && !this.textBlockClosed) {
      output += this.formatEvent({
        type: 'content_block_stop',
        index: 0,
      })
      this.textBlockClosed = true
    }

    // Close thinking block if still open
    if (this.thinkingBlockStarted && !this.thinkingBlockClosed) {
      if (this.thinkingAccumulator) {
        const signature = generateThinkingSignature(this.thinkingAccumulator)
        output += this.formatEvent({
          type: 'content_block_delta',
          index: 1,
          delta: { type: 'signature_delta', signature },
        })
      }
      output += this.formatEvent({
        type: 'content_block_stop',
        index: 1,
      })
      this.thinkingBlockClosed = true
    }

    // Close any open tool blocks
    for (let i = 0; i < this.toolCallBuffers.size; i++) {
      const blockIndex = this.getNextBlockIndex() - this.toolCallBuffers.size + i
      output += this.formatEvent({
        type: 'content_block_stop',
        index: blockIndex,
      })
    }

    // Map finish reason
    const stopReason = mapFinishReasonToStopReason(finishReason)

    // Send message_delta
    output += this.formatEvent({
      type: 'message_delta',
      delta: {
        stop_reason: stopReason,
        stop_sequence: null,
      },
      usage: {
        output_tokens: this.totalOutputTokens,
      },
    })

    // Send message_stop
    output += this.formatEvent({
      type: 'message_stop',
    })

    this.messageStopped = true
    return output
  }

  /**
   * Handle [DONE] marker - finalize the stream
   */
  private handleDone(): string {
    let output = ''

    // Ensure message has started
    output += this.ensureMessageStarted()

    // If not already stopped, finalize
    if (!this.messageStopped) {
      // Close any open blocks
      if (this.textBlockStarted && !this.textBlockClosed) {
        output += this.formatEvent({
          type: 'content_block_stop',
          index: 0,
        })
        this.textBlockClosed = true
      }

      if (this.thinkingBlockStarted && !this.thinkingBlockClosed) {
        if (this.thinkingAccumulator) {
          const signature = generateThinkingSignature(this.thinkingAccumulator)
          output += this.formatEvent({
            type: 'content_block_delta',
            index: 1,
            delta: { type: 'signature_delta', signature },
          })
        }
        output += this.formatEvent({
          type: 'content_block_stop',
          index: 1,
        })
        this.thinkingBlockClosed = true
      }

      // Send message_delta and message_stop
      output += this.formatEvent({
        type: 'message_delta',
        delta: {
          stop_reason: 'end_turn',
          stop_sequence: null,
        },
        usage: {
          output_tokens: this.totalOutputTokens,
        },
      })

      output += this.formatEvent({
        type: 'message_stop',
      })

      this.messageStopped = true
    }

    return output
  }

  /**
   * Finalize the stream (called when stream ends)
   */
  finalize(): string[] {
    const events: string[] = []

    // Ensure message has started
    const startOutput = this.ensureMessageStarted()
    if (startOutput) {
      events.push(startOutput)
    }

    if (!this.messageStopped) {
      // Close any open blocks
      if (this.textBlockStarted && !this.textBlockClosed) {
        events.push(this.formatEvent({
          type: 'content_block_stop',
          index: 0,
        }))
        this.textBlockClosed = true
      }

      if (this.thinkingBlockStarted && !this.thinkingBlockClosed) {
        if (this.thinkingAccumulator) {
          const signature = generateThinkingSignature(this.thinkingAccumulator)
          events.push(this.formatEvent({
            type: 'content_block_delta',
            index: 1,
            delta: { type: 'signature_delta', signature },
          }))
        }
        events.push(this.formatEvent({
          type: 'content_block_stop',
          index: 1,
        }))
        this.thinkingBlockClosed = true
      }

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

      this.messageStopped = true
    }

    return events
  }

  /**
   * Format a Claude stream event as SSE data
   * Each event MUST end with \n\n for proper SSE parsing
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
 * Generate a deterministic signature for thinking blocks
 * Claude Code requires a signature field on thinking blocks for verification.
 * This generates a plausible signature that satisfies the client-side check.
 */
function generateThinkingSignature(thinking: string): string {
  // Generate a deterministic but opaque signature based on the thinking content
  const hash = simpleHash(thinking)
  return `ErUB${hash}`
}

/**
 * Simple deterministic hash function for generating signatures
 */
function simpleHash(str: string): string {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(36).padStart(8, '0') + Math.abs(hash * 31).toString(36).padStart(8, '0')
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
