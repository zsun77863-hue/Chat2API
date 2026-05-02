/**
 * Proxy Service Module - Messages Route
 * Implements Anthropic Messages API at /v1/messages
 * Converts Anthropic format to/from OpenAI format internally
 */

import Router from '@koa/router'
import type { Context } from 'koa'
import { PassThrough } from 'stream'
import { ChatCompletionRequest, ChatCompletionResponse, ProxyContext, ChatMessage } from '../types'
import { loadBalancer } from '../loadbalancer'
import { requestForwarder } from '../forwarder'
import { proxyStatusManager } from '../status'
import { modelMapper } from '../modelMapper'
import { storeManager } from '../../store/store'

const router = new Router({ prefix: '/v1' })

// ─── Anthropic Request Types ────────────────────────────────────────────────

interface AnthropicMessage {
  role: 'user' | 'assistant'
  content: string | AnthropicContentBlock[]
}

interface AnthropicContentBlock {
  type: 'text' | 'tool_use' | 'tool_result' | 'image'
  text?: string
  id?: string
  name?: string
  input?: Record<string, any>
  tool_use_id?: string
  content?: string | any[]
  source?: {
    type: 'base64'
    media_type: string
    data: string
  }
}

interface AnthropicTool {
  name: string
  description?: string
  input_schema: Record<string, any>
}

interface AnthropicToolChoice {
  type: 'auto' | 'any' | 'tool'
  name?: string
  disable_parallel_tool_use?: boolean
}

interface AnthropicRequest {
  model: string
  max_tokens: number
  system?: string | Array<{ type: 'text'; text: string }>
  messages: AnthropicMessage[]
  stream?: boolean
  temperature?: number
  top_p?: number
  stop_sequences?: string[]
  tools?: AnthropicTool[]
  tool_choice?: AnthropicToolChoice
  metadata?: Record<string, any>
}

// ─── Anthropic Response Types ────────────────────────────────────────────────

interface AnthropicTextContent {
  type: 'text'
  text: string
}

interface AnthropicToolUseContent {
  type: 'tool_use'
  id: string
  name: string
  input: Record<string, any>
}

type AnthropicResponseContent = AnthropicTextContent | AnthropicToolUseContent

interface AnthropicUsage {
  input_tokens: number
  output_tokens: number
}

interface AnthropicMessageResponse {
  id: string
  type: 'message'
  role: 'assistant'
  content: AnthropicResponseContent[]
  model: string
  stop_reason: 'end_turn' | 'max_tokens' | 'stop_sequence' | 'tool_use' | null
  stop_sequence: string | null
  usage: AnthropicUsage
}

// ─── Helper Functions ────────────────────────────────────────────────────────

function generateAnthropicMessageId(): string {
  return `msg_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`
}

function getClientIP(ctx: Context): string {
  return (ctx.headers['x-real-ip'] as string) ||
    (ctx.headers['x-forwarded-for'] as string) ||
    ctx.ip ||
    'unknown'
}

/**
 * Extract user input from Anthropic messages (last user message content)
 */
function extractUserInput(messages: AnthropicMessage[]): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role === 'user' && msg.content) {
      if (typeof msg.content === 'string') {
        return msg.content
      }
      if (Array.isArray(msg.content)) {
        const textParts = msg.content.filter((b) => b.type === 'text' && b.text)
        if (textParts.length > 0) {
          return textParts.map((b) => b.text!).join(' ')
        }
      }
    }
  }
  return undefined
}

/**
 * Convert a single Anthropic content block array to OpenAI content format
 */
function anthropicContentToOpenAI(
  content: string | AnthropicContentBlock[]
): string | Array<{ type: string; text?: string; image_url?: { url: string; detail?: string }; tool_call_id?: string; content?: string }> {
  if (typeof content === 'string') {
    return content
  }

  const result: Array<any> = []
  for (const block of content) {
    if (block.type === 'text' && block.text) {
      result.push({ type: 'text', text: block.text })
    } else if (block.type === 'image' && block.source) {
      result.push({
        type: 'image_url',
        image_url: {
          url: `data:${block.source.media_type};base64,${block.source.data}`,
        },
      })
    }
  }

  // If all blocks are text, simplify to plain string
  if (result.length === 1 && result[0].type === 'text') {
    return result[0].text
  }
  if (result.length === 0) {
    return ''
  }
  return result
}

/**
 * Convert Anthropic request → internal ChatCompletionRequest
 */
function convertAnthropicToChatCompletion(anthropicReq: AnthropicRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = []

  // 1. Prepend system message if present
  if (anthropicReq.system) {
    if (typeof anthropicReq.system === 'string') {
      messages.push({ role: 'system', content: anthropicReq.system })
    } else if (Array.isArray(anthropicReq.system)) {
      // system can be an array of {type: 'text', text: '...'} blocks
      const textParts = anthropicReq.system
        .filter((b) => b.type === 'text' && b.text)
        .map((b) => b.text!)
        .join('\n')
      if (textParts) {
        messages.push({ role: 'system', content: textParts })
      }
    }
  }

  // 2. Convert messages
  for (const msg of anthropicReq.messages) {
    if (msg.role !== 'user' && msg.role !== 'assistant') continue

    if (typeof msg.content === 'string') {
      // Simple string content
      messages.push({ role: msg.role, content: msg.content })
      continue
    }

    if (!Array.isArray(msg.content)) continue

    // Separate tool_result blocks from other content
    const toolResults = msg.content.filter((b) => b.type === 'tool_result')
    const nonToolBlocks = msg.content.filter((b) => b.type !== 'tool_result') as AnthropicContentBlock[]

    // Send tool results as individual tool messages
    for (const tr of toolResults) {
      let toolContent = ''
      if (typeof tr.content === 'string') {
        toolContent = tr.content
      } else if (Array.isArray(tr.content)) {
        const textParts = tr.content.filter((b: any) => b.type === 'text')
        toolContent = textParts.map((b: any) => b.text || '').join('\n')
      }
      messages.push({
        role: 'tool',
        content: toolContent,
        tool_call_id: tr.tool_use_id,
      })
    }

    // Process non-tool content blocks
    if (nonToolBlocks.length === 0) continue

    // Check for tool_use blocks in assistant content → convert to tool_calls
    const toolUseBlocks = nonToolBlocks.filter((b) => b.type === 'tool_use')
    const nonToolUseBlocks = nonToolBlocks.filter((b) => b.type !== 'tool_use')

    if (msg.role === 'assistant' && toolUseBlocks.length > 0) {
      // Assistant message with tool_use blocks
      const textContent = nonToolUseBlocks
        .filter((b) => b.type === 'text' && b.text)
        .map((b) => b.text!)
        .join('')

      const toolCalls = toolUseBlocks.map((tu, idx) => ({
        id: tu.id || `call_${Date.now().toString(36)}_${idx}`,
        type: 'function' as const,
        function: {
          name: tu.name || '',
          arguments: JSON.stringify(tu.input || {}),
        },
      }))

      messages.push({
        role: 'assistant',
        content: textContent || null,
        tool_calls: toolCalls,
      })
    } else if (nonToolUseBlocks.length > 0) {
      // Regular content (text/image blocks) → single message
      const converted = anthropicContentToOpenAI(nonToolUseBlocks)
      messages.push({ role: msg.role, content: converted as any })
    }
  }

  // 3. Convert tools
  const tools = anthropicReq.tools?.map((tool) => ({
    type: 'function' as const,
    function: {
      name: tool.name,
      description: tool.description || '',
      parameters: tool.input_schema,
    },
  }))

  // 4. Convert tool_choice
  let toolChoice: ChatCompletionRequest['tool_choice']
  if (anthropicReq.tool_choice) {
    switch (anthropicReq.tool_choice.type) {
      case 'auto':
        toolChoice = 'auto'
        break
      case 'any':
        // Anthropic "any" = force a tool call → OpenAI "required"
        if (anthropicReq.tool_choice.name) {
          toolChoice = {
            type: 'function',
            function: { name: anthropicReq.tool_choice.name },
          }
        } else {
          toolChoice = 'required'
        }
        break
      case 'tool':
        // Force a specific tool
        if (anthropicReq.tool_choice.name) {
          toolChoice = {
            type: 'function',
            function: { name: anthropicReq.tool_choice.name },
          }
        }
        break
    }
  }

  return {
    model: anthropicReq.model,
    messages,
    stream: anthropicReq.stream || false,
    temperature: anthropicReq.temperature,
    top_p: anthropicReq.top_p,
    max_tokens: anthropicReq.max_tokens,
    stop: anthropicReq.stop_sequences,
    tools,
    tool_choice: toolChoice,
  }
}

/**
 * Map OpenAI finish_reason → Anthropic stop_reason
 */
function mapStopReason(finishReason: string | null): AnthropicMessageResponse['stop_reason'] {
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
      return null
  }
}

/**
 * Convert OpenAI non-streaming response → Anthropic MessageResponse
 */
function convertResponseToAnthropic(
  openaiResponse: ChatCompletionResponse,
  requestId: string,
  model: string
): AnthropicMessageResponse {
  const choice = openaiResponse.choices?.[0]
  const message = choice?.message
  const finishReason = choice?.finish_reason || null

  const content: AnthropicResponseContent[] = []

  if (message?.content) {
    content.push({ type: 'text', text: message.content })
  }

  // Convert tool_calls to tool_use content blocks
  if (message?.tool_calls && message.tool_calls.length > 0) {
    for (const tc of message.tool_calls) {
      let input: Record<string, any> = {}
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {
        // If arguments aren't valid JSON, wrap as string
        input = { raw: tc.function.arguments }
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      })
    }
  }

  // If no content at all, provide empty text block
  if (content.length === 0) {
    content.push({ type: 'text', text: '' })
  }

  return {
    id: requestId,
    type: 'message',
    role: 'assistant',
    content,
    model: model,
    stop_reason: mapStopReason(finishReason),
    stop_sequence: null,
    usage: {
      input_tokens: openaiResponse.usage?.prompt_tokens || 0,
      output_tokens: openaiResponse.usage?.completion_tokens || 0,
    },
  }
}

/**
 * Format an Anthropic SSE event string
 */
function formatAnthropicSSE(eventName: string, data: object): string {
  return `event: ${eventName}\ndata: ${JSON.stringify(data)}\n\n`
}

// ─── Route Handler ──────────────────────────────────────────────────────────

router.post('/messages', async (ctx: Context) => {
  const startTime = Date.now()
  const requestId = generateAnthropicMessageId()
  const clientIP = getClientIP(ctx)

  // Parse request body
  let anthropicReq: AnthropicRequest
  try {
    anthropicReq = ctx.request.body as AnthropicRequest
  } catch {
    ctx.status = 400
    ctx.body = {
      type: 'error',
      error: {
        type: 'invalid_request_error',
        message: 'Invalid request body',
      },
    }
    return
  }

  // Validate required fields
  if (!anthropicReq.model) {
    ctx.status = 400
    ctx.body = {
      type: 'error',
      error: {
        type: 'invalid_request_error',
        message: 'Missing required field: model',
      },
    }
    return
  }

  if (!anthropicReq.max_tokens || anthropicReq.max_tokens <= 0) {
    ctx.status = 400
    ctx.body = {
      type: 'error',
      error: {
        type: 'invalid_request_error',
        message: 'Missing or invalid required field: max_tokens',
      },
    }
    return
  }

  if (!anthropicReq.messages || !Array.isArray(anthropicReq.messages) || anthropicReq.messages.length === 0) {
    ctx.status = 400
    ctx.body = {
      type: 'error',
      error: {
        type: 'invalid_request_error',
        message: 'Missing required field: messages',
      },
    }
    return
  }

  // Read feature parameters from Headers
  const webSearchFromHeader = ctx.headers['x-web-search'] === 'true'
  const reasoningEffortFromHeader = ctx.headers['x-reasoning-effort'] as 'low' | 'medium' | 'high' | undefined

  // Convert Anthropic request → OpenAI format
  const chatRequest = convertAnthropicToChatCompletion(anthropicReq)

  // Apply header-based feature flags
  if (webSearchFromHeader && chatRequest.web_search === undefined) {
    chatRequest.web_search = true
    console.log('[Messages] Web search enabled via X-Web-Search header')
  }
  if (reasoningEffortFromHeader && chatRequest.reasoning_effort === undefined) {
    chatRequest.reasoning_effort = reasoningEffortFromHeader
    console.log('[Messages] Reasoning effort set via X-Reasoning-Effort header:', reasoningEffortFromHeader)
  }

  // Account selection
  const config = storeManager.getConfig()
  const preferredProviderId = modelMapper.getPreferredProvider(chatRequest.model)
  const preferredAccountId = modelMapper.getPreferredAccount(chatRequest.model)

  const selection = loadBalancer.selectAccount(
    chatRequest.model,
    config.loadBalanceStrategy,
    preferredProviderId,
    preferredAccountId
  )

  if (!selection) {
    ctx.status = 503
    ctx.body = {
      type: 'error',
      error: {
        type: 'service_unavailable_error',
        message: `No available account for model: ${chatRequest.model}`,
      },
    }
    return
  }

  const { account, provider, actualModel } = selection

  const context: ProxyContext = {
    requestId,
    providerId: provider.id,
    accountId: account.id,
    model: chatRequest.model,
    actualModel,
    startTime,
    isStream: anthropicReq.stream || false,
    clientIP,
  }

  proxyStatusManager.recordRequestStart(chatRequest.model, provider.id, account.id)

  try {
    const result = await requestForwarder.forwardChatCompletion(
      chatRequest,
      account,
      provider,
      actualModel,
      context
    )

    const latency = Date.now() - startTime

    // ── Error Handling ─────────────────────────────────────────────────────
    if (!result.success) {
      proxyStatusManager.recordRequestFailure(latency)

      if (result.status && result.status >= 400 && result.status !== 429) {
        loadBalancer.markAccountFailed(account.id)
      }

      ctx.status = result.status || 500
      ctx.body = {
        type: 'error',
        error: {
          type: 'api_error',
          message: result.error || 'Request failed',
        },
      }

      storeManager.addLog('error', `[Messages] Request failed: ${result.error}`, {
        requestId,
        providerId: provider.id,
        accountId: account.id,
        model: chatRequest.model,
        latency,
      })

      const userInput = extractUserInput(anthropicReq.messages)
      storeManager.addRequestLog({
        timestamp: startTime,
        status: 'error',
        statusCode: result.status || 500,
        method: 'POST',
        url: '/v1/messages',
        model: chatRequest.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(anthropicReq),
        userInput,
        responseStatus: result.status || 500,
        responseBody: JSON.stringify(ctx.body),
        latency,
        isStream: anthropicReq.stream || false,
        errorMessage: result.error,
      })

      storeManager.recordRequestInStats(false, latency, chatRequest.model, provider.id, account.id)
      return
    }

    // ── Success Path ───────────────────────────────────────────────────────
    loadBalancer.clearAccountFailure(account.id)
    proxyStatusManager.recordRequestSuccess(latency)

    storeManager.updateAccount(account.id, {
      lastUsed: Date.now(),
      requestCount: (account.requestCount || 0) + 1,
      todayUsed: (account.todayUsed || 0) + 1,
    })

    storeManager.addLog('debug', `[Messages] Request succeeded`, {
      requestId,
      providerId: provider.id,
      accountId: account.id,
      model: chatRequest.model,
      actualModel,
      latency,
      isStream: anthropicReq.stream,
    })

    const userInput = extractUserInput(anthropicReq.messages)

    // ── Streaming Response ─────────────────────────────────────────────────
    if (anthropicReq.stream === true && result.stream) {
      ctx.set('Content-Type', 'text/event-stream')
      ctx.set('Cache-Control', 'no-cache')
      ctx.set('Connection', 'keep-alive')
      ctx.set('X-Accel-Buffering', 'no')

      const anthropicStream = new PassThrough()
      let collectedContent = ''
      let contentBlockStarted = false
      let contentBlockIndex = 0
      // Buffer for incomplete SSE lines from upstream
      let sseBuffer = ''
      // Track tool calls across streaming chunks
      let toolCallBuffer: Map<number, { id: string; name: string; arguments: string }> = new Map()
      let sentToolUseBlocks = new Set<number>()

      // Send message_start event
      const messageStart = {
        type: 'message_start',
        message: {
          id: requestId,
          type: 'message',
          role: 'assistant',
          content: [],
          model: actualModel,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }
      anthropicStream.write(formatAnthropicSSE('message_start', messageStart))

      // Send initial ping
      anthropicStream.write(formatAnthropicSSE('ping', { type: 'ping' }))

      // Collect log entry
      let logEntryId: string | undefined
      const logEntry = storeManager.addRequestLog({
        timestamp: startTime,
        status: 'success',
        statusCode: 200,
        method: 'POST',
        url: '/v1/messages',
        model: chatRequest.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(anthropicReq),
        userInput,
        responseStatus: 200,
        latency,
        isStream: true,
      })
      logEntryId = logEntry.id

      storeManager.recordRequestInStats(true, latency, chatRequest.model, provider.id, account.id)

      // Helper: ensure a content_block_start has been sent
      const ensureContentBlockStarted = (blockType: 'text' | 'tool_use', toolInfo?: { id: string; name: string }) => {
        if (!contentBlockStarted) {
          const startEvent: any = {
            type: 'content_block_start',
            index: contentBlockIndex,
            content_block: blockType === 'text'
              ? { type: 'text', text: '' }
              : { type: 'tool_use', id: toolInfo?.id || '', name: toolInfo?.name || '', input: {} },
          }
          anthropicStream.write(formatAnthropicSSE('content_block_start', startEvent))
          contentBlockStarted = true
        }
      }

      // Helper: close current content block
      const closeContentBlock = () => {
        if (contentBlockStarted) {
          anthropicStream.write(formatAnthropicSSE('content_block_stop', {
            type: 'content_block_stop',
            index: contentBlockIndex,
          }))
          contentBlockIndex++
          contentBlockStarted = false
        }
      }

      // Process a single parsed OpenAI SSE chunk
      const processOpenAIChunk = (data: any) => {
        const choice = data.choices?.[0]
        if (!choice) return

        const delta = choice.delta
        const finishReason = choice.finish_reason

        // Handle text content
        if (delta?.content) {
          ensureContentBlockStarted('text')
          collectedContent += delta.content
          anthropicStream.write(formatAnthropicSSE('content_block_delta', {
            type: 'content_block_delta',
            index: contentBlockIndex,
            delta: { type: 'text_delta', text: delta.content },
          }))
        }

        // Handle tool_calls deltas
        if (delta?.tool_calls) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index ?? 0
            if (!toolCallBuffer.has(idx)) {
              toolCallBuffer.set(idx, { id: tc.id || '', name: tc.function?.name || '', arguments: '' })
            }
            const buf = toolCallBuffer.get(idx)!
            if (tc.id) buf.id = tc.id
            if (tc.function?.name) buf.name = tc.function.name
            if (tc.function?.arguments) buf.arguments += tc.function.arguments
          }
        }

        // Handle finish_reason
        if (finishReason) {
          // If tool calls were accumulated, emit tool_use blocks
          if (toolCallBuffer.size > 0) {
            for (const [idx, tc] of toolCallBuffer) {
              if (sentToolUseBlocks.has(idx)) continue

              // Close any open text block first
              closeContentBlock()

              // Parse accumulated arguments
              let parsedInput: Record<string, any> = {}
              try {
                parsedInput = JSON.parse(tc.arguments)
              } catch {
                parsedInput = { raw: tc.arguments }
              }

              // Start tool_use content block
              const startEvent = {
                type: 'content_block_start',
                index: contentBlockIndex,
                content_block: {
                  type: 'tool_use',
                  id: tc.id,
                  name: tc.name,
                  input: {},
                },
              }
              anthropicStream.write(formatAnthropicSSE('content_block_start', startEvent))

              // Send the full input as a single delta
              anthropicStream.write(formatAnthropicSSE('content_block_delta', {
                type: 'content_block_delta',
                index: contentBlockIndex,
                delta: { type: 'input_json_delta', partial_json: JSON.stringify(parsedInput) },
              }))

              // Close the tool_use block
              anthropicStream.write(formatAnthropicSSE('content_block_stop', {
                type: 'content_block_stop',
                index: contentBlockIndex,
              }))
              contentBlockIndex++
              sentToolUseBlocks.add(idx)
            }
          }

          // Close any open text block
          closeContentBlock()

          // Send message_delta with stop_reason
          const stopReason = mapStopReason(finishReason)
          anthropicStream.write(formatAnthropicSSE('message_delta', {
            type: 'message_delta',
            delta: {
              stop_reason: stopReason,
              stop_sequence: null,
            },
            usage: { output_tokens: data.usage?.completion_tokens || 0 },
          }))

          // Send message_stop
          anthropicStream.write(formatAnthropicSSE('message_stop', { type: 'message_stop' }))
        }
      }

      // Handle upstream stream errors
      result.stream.once('error', (err: Error) => {
        console.error('[Messages] Stream error:', err.message)

        // Close any open content block
        closeContentBlock()

        // Send error as an Anthropic error event
        anthropicStream.write(formatAnthropicSSE('error', {
          type: 'error',
          error: {
            type: 'api_error',
            message: err.message,
          },
        }))

        anthropicStream.end()

        storeManager.addLog('error', `[Messages] Stream error: ${err.message}`, {
          requestId,
          providerId: provider.id,
          accountId: account.id,
          model: chatRequest.model,
        })
      })

      // Parse upstream SSE data
      result.stream.on('data', (chunk: Buffer) => {
        sseBuffer += chunk.toString()
        const lines = sseBuffer.split('\n')
        // Keep the last potentially incomplete line in buffer
        sseBuffer = lines.pop() || ''

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue

          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6)

            if (dataStr === '[DONE]') {
              // If message_stop hasn't been sent yet (e.g. upstream sent [DONE]
              // without a finish_reason chunk), send closing events now.
              if (contentBlockStarted) {
                closeContentBlock()
              }
              // message_delta + message_stop may already have been sent by
              // processOpenAIChunk when it saw finish_reason. Guard against
              // double-sending by checking if stream is still writable.
              if (!anthropicStream.writableEnded) {
                anthropicStream.write(formatAnthropicSSE('message_delta', {
                  type: 'message_delta',
                  delta: { stop_reason: 'end_turn', stop_sequence: null },
                  usage: { output_tokens: 0 },
                }))
                anthropicStream.write(formatAnthropicSSE('message_stop', { type: 'message_stop' }))
                anthropicStream.end()
              }
              continue
            }

            try {
              const data = JSON.parse(dataStr)
              processOpenAIChunk(data)
            } catch {
              // Ignore unparseable chunks
            }
          }
          // Ignore other SSE fields (event:, id:, etc.) from upstream
        }
      })

      // Handle stream end
      result.stream.on('end', () => {
        // Process any remaining buffer
        if (sseBuffer.trim()) {
          const trimmed = sseBuffer.trim()
          if (trimmed.startsWith('data: ') && trimmed.slice(6) !== '[DONE]') {
            try {
              const data = JSON.parse(trimmed.slice(6))
              processOpenAIChunk(data)
            } catch {
              // Ignore
            }
          }
        }

        // If stream ended without explicit [DONE], ensure we close properly
        if (!anthropicStream.writableEnded) {
          if (contentBlockStarted) {
            closeContentBlock()
          }
          anthropicStream.write(formatAnthropicSSE('message_delta', {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: 0 },
          }))
          anthropicStream.write(formatAnthropicSSE('message_stop', { type: 'message_stop' }))
          anthropicStream.end()
        }

        // Update log
        if (logEntryId) {
          storeManager.updateRequestLog(logEntryId, {
            responseBody: collectedContent || undefined,
          })
        }
      })

      ctx.body = anthropicStream
    } else {
      // ── Non-Streaming Response ───────────────────────────────────────────
      ctx.set('Content-Type', 'application/json')

      if (result.body) {
        const anthropicResponse = convertResponseToAnthropic(
          result.body as ChatCompletionResponse,
          requestId,
          actualModel
        )
        ctx.body = anthropicResponse
      } else {
        // Fallback empty response
        ctx.body = {
          id: requestId,
          type: 'message',
          role: 'assistant',
          content: [{ type: 'text', text: '' }],
          model: actualModel,
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        }
      }

      storeManager.addRequestLog({
        timestamp: startTime,
        status: 'success',
        statusCode: 200,
        method: 'POST',
        url: '/v1/messages',
        model: chatRequest.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(anthropicReq),
        userInput,
        responseStatus: 200,
        responseBody: JSON.stringify(ctx.body),
        latency,
        isStream: false,
      })

      storeManager.recordRequestInStats(true, latency, chatRequest.model, provider.id, account.id)
    }
  } catch (error) {
    const latency = Date.now() - startTime
    proxyStatusManager.recordRequestFailure(latency)

    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    const errorStack = error instanceof Error ? error.stack : undefined

    ctx.status = 500
    ctx.body = {
      type: 'error',
      error: {
        type: 'internal_error',
        message: errorMessage,
      },
    }

    storeManager.addLog('error', `[Messages] Request exception: ${errorMessage}`, {
      requestId,
      providerId: provider.id,
      accountId: account.id,
      model: chatRequest.model,
      latency,
      error: errorMessage,
    })

    const userInput = extractUserInput(anthropicReq.messages)
    storeManager.addRequestLog({
      timestamp: startTime,
      status: 'error',
      statusCode: 500,
      method: 'POST',
      url: '/v1/messages',
      model: chatRequest.model,
      actualModel,
      providerId: provider.id,
      providerName: provider.name,
      accountId: account.id,
      accountName: account.name,
      requestBody: JSON.stringify(anthropicReq),
      userInput,
      responseStatus: 500,
      responseBody: JSON.stringify(ctx.body),
      latency,
      isStream: anthropicReq.stream || false,
      errorMessage,
      errorStack,
    })

    storeManager.recordRequestInStats(false, latency, chatRequest.model, provider.id, account.id)
  }
})

export default router
