/**
 * Proxy Service Module - Chat Completions Route
 * Implements /v1/chat/completions route
 * 
 * v1.7.2 改进：
 * - Agent 循环改为实时流式推送，废除批量缓存一次性输出
 * - 每一步思考、工具调用、结果都立刻流式推送给用户
 * - 隐藏内部系统标记
 * - 工具调用流程拆分流式化
 */

import Router from '@koa/router'
import type { Context } from 'koa'
import { PassThrough } from 'stream'
import { ChatCompletionRequest, ChatCompletionResponse, ProxyContext } from '../types'
import { loadBalancer } from '../loadbalancer'
import { requestForwarder } from '../forwarder'
import { streamHandler } from '../stream'
import { proxyStatusManager } from '../status'
import { modelMapper } from '../modelMapper'
import { storeManager } from '../../store/store'
import { 
  isAnthropicToolFormat,
  transformResponseToAnthropic,
  transformChunkToAnthropic
} from '../utils/toolFormatConverter'
import { agentLoopManager, filterInternalMarkers, isOnlyInternalMarkers } from '../agentLoop'

const router = new Router({ prefix: '/v1/chat' })

/**
 * Generate Request ID
 */
function generateRequestId(): string {
  return `chatcmpl-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

/**
 * Get Client IP
 */
function getClientIP(ctx: Context): string {
  return ctx.headers['x-real-ip'] as string ||
    ctx.headers['x-forwarded-for'] as string ||
    ctx.ip ||
    'unknown'
}

/**
 * Extract user input from messages (last user message, full content)
 */
function extractUserInput(messages: Array<{ role: string; content?: string | any[] | null }>): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role === 'user' && msg.content) {
      let content = ''
      if (typeof msg.content === 'string') {
        content = msg.content
      } else if (Array.isArray(msg.content)) {
        const textParts = msg.content.filter((p: any) => p.type === 'text')
        if (textParts.length > 0) {
          content = textParts.map((p: any) => p.text || '').join(' ')
        }
      }
      if (content) {
        return content
      }
    }
  }
  return undefined
}

/**
 * 写入 SSE 格式的 chunk 到流
 */
function writeSSEChunk(
  stream: PassThrough,
  requestId: string,
  actualModel: string,
  delta: Record<string, any>,
  finishReason: string | null = null
): void {
  stream.write(`data: ${JSON.stringify({
    id: requestId,
    object: 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model: actualModel,
    choices: [{
      index: 0,
      delta,
      finish_reason: finishReason,
    }],
  })}\n\n`)
}

/**
 * 写入 Agent 步骤分隔标记（视觉分隔，让用户能看清每一步）
 */
function writeAgentStepSeparator(
  stream: PassThrough,
  requestId: string,
  actualModel: string,
  stepInfo: string
): void {
  const separator = `\n\n---\n**🔄 ${stepInfo}**\n\n`
  writeSSEChunk(stream, requestId, actualModel, { content: separator })
}

/**
 * 处理流式响应，实时提取内容和 tool_calls
 * 返回收集到的完整内容、tool_calls 和 reasoning_content
 */
async function processStreamInRealTime(
  sourceStream: NodeJS.ReadableStream,
  sseStream: PassThrough,
  requestId: string,
  actualModel: string,
  skipTransform: boolean
): Promise<{
  content: string
  toolCalls: any[]
  reasoningContent: string
}> {
  return new Promise((resolve, reject) => {
    let collectedContent = ''
    let collectedReasoning = ''
    const toolCallMap = new Map<number, { id: string; name: string; arguments: string }>()
    let buffer = ''

    sourceStream.on('data', (chunk: Buffer) => {
      const text = chunk.toString()
      buffer += text

      // 按行解析 SSE 数据
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // 保留最后一行（可能不完整）

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed || trimmed.startsWith(':')) continue

        if (trimmed.startsWith('data: ')) {
          const data = trimmed.slice(6)
          if (data === '[DONE]') continue

          try {
            const parsed = JSON.parse(data)
            const choice = parsed.choices?.[0]
            if (!choice) continue

            const delta = choice.delta

            // 提取 reasoning_content
            if (delta?.reasoning_content) {
              const filtered = filterInternalMarkers(delta.reasoning_content)
              if (filtered) {
                collectedReasoning += filtered
                // 实时推送 reasoning_content 给客户端
                writeSSEChunk(sseStream, requestId, actualModel, {
                  reasoning_content: filtered
                })
              }
            }

            // 提取 content
            if (delta?.content) {
              const filtered = filterInternalMarkers(delta.content)
              if (filtered) {
                collectedContent += filtered
                // 实时推送 content 给客户端
                writeSSEChunk(sseStream, requestId, actualModel, {
                  content: filtered
                })
              }
            }

            // 提取 tool_calls
            if (delta?.tool_calls) {
              for (const tc of delta.tool_calls) {
                const idx = tc.index ?? 0
                if (!toolCallMap.has(idx)) {
                  toolCallMap.set(idx, {
                    id: tc.id || '',
                    name: tc.function?.name || '',
                    arguments: '',
                  })
                }
                const existing = toolCallMap.get(idx)!
                if (tc.id) existing.id = tc.id
                if (tc.function?.name) existing.name = tc.function.name
                if (tc.function?.arguments) existing.arguments += tc.function.arguments

                // 实时推送 tool_calls 给客户端
                writeSSEChunk(sseStream, requestId, actualModel, {
                  tool_calls: [tc]
                })
              }
            }

            // 检查 finish_reason
            if (choice.finish_reason) {
              // 不在这里结束流，因为 Agent 可能还有后续轮次
            }
          } catch {
            // 跳过无法解析的行
          }
        }
      }
    })

    sourceStream.on('end', () => {
      // 处理 buffer 中剩余的数据
      if (buffer.trim()) {
        const trimmed = buffer.trim()
        if (trimmed.startsWith('data: ') && trimmed.slice(6) !== '[DONE]') {
          try {
            const parsed = JSON.parse(trimmed.slice(6))
            const choice = parsed.choices?.[0]
            if (choice) {
              const delta = choice.delta
              if (delta?.content) {
                const filtered = filterInternalMarkers(delta.content)
                if (filtered) {
                  collectedContent += filtered
                  writeSSEChunk(sseStream, requestId, actualModel, { content: filtered })
                }
              }
              if (delta?.reasoning_content) {
                const filtered = filterInternalMarkers(delta.reasoning_content)
                if (filtered) {
                  collectedReasoning += filtered
                  writeSSEChunk(sseStream, requestId, actualModel, { reasoning_content: filtered })
                }
              }
            }
          } catch {
            // 忽略
          }
        }
      }

      const toolCalls = Array.from(toolCallMap.entries())
        .sort(([a], [b]) => a - b)
        .map(([index, tc]) => ({
          index,
          id: tc.id,
          type: 'function' as const,
          function: {
            name: tc.name,
            arguments: tc.arguments,
          },
        }))

      resolve({
        content: collectedContent,
        toolCalls,
        reasoningContent: collectedReasoning,
      })
    })

    sourceStream.on('error', (err: Error) => {
      reject(err)
    })
  })
}

/**
 * Handle Chat Completions Request
 */
router.post('/completions', async (ctx: Context) => {
  const startTime = Date.now()
  const requestId = generateRequestId()
  const clientIP = getClientIP(ctx)

  let request: ChatCompletionRequest
  try {
    request = ctx.request.body as ChatCompletionRequest
  } catch (error) {
    ctx.status = 400
    ctx.body = {
      error: {
        message: 'Invalid request body',
        type: 'invalid_request_error',
        param: null,
        code: null,
      },
    }
    return
  }

  if (!request.model) {
    ctx.status = 400
    ctx.body = {
      error: {
        message: 'Missing required field: model',
        type: 'invalid_request_error',
        param: 'model',
        code: null,
      },
    }
    return
  }

  if (!request.messages || !Array.isArray(request.messages) || request.messages.length === 0) {
    ctx.status = 400
    ctx.body = {
      error: {
        message: 'Missing required field: messages',
        type: 'invalid_request_error',
        param: 'messages',
        code: null,
      },
    }
    return
  }

  // Read feature parameters from Headers (lower priority than request body)
  const webSearchFromHeader = ctx.headers['x-web-search'] === 'true'
  const reasoningEffortFromHeader = ctx.headers['x-reasoning-effort'] as 'low' | 'medium' | 'high' | undefined
  const deepResearchFromHeader = ctx.headers['x-deep-research'] === 'true'

  // Handle reasoningEffort (camelCase) from AI SDK - convert to reasoning_effort (snake_case)
  const requestAny = request as any
  if (requestAny.reasoningEffort && !request.reasoning_effort) {
    request.reasoning_effort = requestAny.reasoningEffort
    console.log('[Chat] Reasoning effort set via reasoningEffort (camelCase):', requestAny.reasoningEffort)
    delete requestAny.reasoningEffort
  }

  // Merge into request (request body parameters take priority)
  if (webSearchFromHeader && request.web_search === undefined) {
    request.web_search = true
    console.log('[Chat] Web search enabled via X-Web-Search header')
  }
  if (reasoningEffortFromHeader && request.reasoning_effort === undefined) {
    request.reasoning_effort = reasoningEffortFromHeader
    console.log('[Chat] Reasoning effort set via X-Reasoning-Effort header:', reasoningEffortFromHeader)
  }
  if (deepResearchFromHeader && request.deep_research === undefined) {
    request.deep_research = true
    console.log('[Chat] Deep research enabled via X-Deep-Research header')
  }

  const config = storeManager.getConfig()
  const preferredProviderId = modelMapper.getPreferredProvider(request.model)
  const preferredAccountId = modelMapper.getPreferredAccount(request.model)

  const selection = loadBalancer.selectAccount(
    request.model,
    config.loadBalanceStrategy,
    preferredProviderId,
    preferredAccountId
  )

  if (!selection) {
    ctx.status = 503
    ctx.body = {
      error: {
        message: `No available account for model: ${request.model}`,
        type: 'service_unavailable_error',
        param: null,
        code: 'no_available_account',
      },
    }
    return
  }

  const { account, provider, actualModel } = selection

  const context: ProxyContext = {
    requestId,
    providerId: provider.id,
    accountId: account.id,
    model: request.model,
    actualModel,
    startTime,
    isStream: request.stream || false,
    clientIP,
  }

  proxyStatusManager.recordRequestStart(request.model, provider.id, account.id)

  // ============================================================
  // Agent Loop Mode - 多轮工具调用自动循环（默认自动启用）
  // v1.7.2: 改为实时流式推送，废除批量缓存
  // ============================================================
  const agentModeEnabled = agentLoopManager.shouldEnable(request, ctx.headers as any)
  let agentSessionId: string | undefined
  const clientWantsStream = request.stream === true

  if (agentModeEnabled) {
    const isCont = agentLoopManager.isContinuation(request, ctx.headers as any)

    if (isCont) {
      // ---- 续接模式：前端回传了工具执行结果 ----
      agentSessionId = agentLoopManager.resolveSessionId(request, ctx.headers as any)
      const session = agentSessionId ? agentLoopManager.appendToolResults(agentSessionId, request) : null

      if (!session || !agentSessionId) {
        console.warn('[Chat] Agent session not found for continuation, falling back to normal mode')
        agentSessionId = undefined
      } else if (agentLoopManager.isMaxRoundsExceeded(agentSessionId)) {
        const maxR = agentLoopManager.getMaxRounds(agentSessionId)
        console.warn(`[Chat] Agent max rounds exceeded (${maxR}), returning error`)
        proxyStatusManager.recordRequestFailure(Date.now() - startTime)
        ctx.status = 400
        ctx.set('X-Agent-Loop', 'max_rounds_exceeded')
        ctx.set('X-Agent-Round', String(maxR))
        ctx.body = {
          error: {
            message: `Agent loop exceeded maximum rounds (${maxR}). The task may be too complex or there may be an infinite tool calling loop.`,
            type: 'agent_loop_error',
            param: null,
            code: 'max_rounds_exceeded',
          },
        }
        agentLoopManager.deleteSession(agentSessionId)
        return
      } else {
        // 使用会话中缓存的完整消息历史构建下一轮请求
        const nextRequest = agentLoopManager.buildNextStreamRequest(session, request)
        request.messages = nextRequest.messages
        if (nextRequest.tools) request.tools = nextRequest.tools
        if (nextRequest.tool_choice !== undefined) request.tool_choice = nextRequest.tool_choice
        console.log(`[Chat] Agent continuation: session=${agentSessionId}, round=${agentLoopManager.getRoundCount(agentSessionId)}, messages=${request.messages.length}`)
      }
    } else {
      // ---- 新会话模式 ----
      agentSessionId = agentLoopManager.generateSessionId()
      agentLoopManager.createSession(agentSessionId, request)
      console.log(`[Chat] Agent new session created: ${agentSessionId}`)
    }

    // v1.7.2: Agent 循环使用流式请求，实时推送每一步
    // 不再强制 stream: false，而是保持客户端原始的 stream 设置
    // 这样每一步的输出都能实时推送给用户
    request.stream = true
  }

  try {
    // ============================================================
    // v1.7.2: Agent 模式 + 客户端要求流式 → 实时流式 Agent 循环
    // ============================================================
    if (agentModeEnabled && clientWantsStream) {
      // 设置 SSE 响应头
      ctx.set('Content-Type', 'text/event-stream')
      ctx.set('Cache-Control', 'no-cache')
      ctx.set('Connection', 'keep-alive')
      ctx.set('X-Accel-Buffering', 'no')
      ctx.set('X-Session-Id', agentSessionId || '')

      const sseStream = new PassThrough()
      ctx.body = sseStream

      // 发送第一个 chunk: role
      writeSSEChunk(sseStream, requestId, actualModel, { role: 'assistant' })

      // Agent 循环：实时流式推送
      let currentRound = 0
      const maxRounds = agentSessionId ? agentLoopManager.getMaxRounds(agentSessionId) : 50
      let allCollectedContent = ''
      let allCollectedReasoning = ''

      // 保存原始请求信息用于后续轮次
      const originalRequest = { ...request }

      while (currentRound < maxRounds) {
        currentRound++
        console.log(`[Chat] Agent round ${currentRound}/${maxRounds} starting...`)

        // 写入步骤分隔标记
        if (currentRound > 1) {
          writeAgentStepSeparator(sseStream, requestId, actualModel, `Round ${currentRound}`)
        }

        // 发起流式请求
        const result = await requestForwarder.forwardChatCompletion(
          request,
          account,
          provider,
          actualModel,
          context
        )

        if (!result.success) {
          // 请求失败，推送错误信息
          const errorMsg = result.error || 'Request failed in agent loop'
          writeSSEChunk(sseStream, requestId, actualModel, {
            content: `\n\n❌ **Error in round ${currentRound}:** ${errorMsg}\n`
          })
          writeSSEChunk(sseStream, requestId, actualModel, {}, 'stop')
          sseStream.write('data: [DONE]\n\n')
          sseStream.end()
          break
        }

        if (!result.stream) {
          // 非流式结果（不应该发生，因为我们设置了 stream: true）
          // 但作为降级处理
          const body = result.body
          if (body) {
            const message = body.choices?.[0]?.message
            if (message?.reasoning_content) {
              const filtered = filterInternalMarkers(message.reasoning_content)
              if (filtered) {
                allCollectedReasoning += filtered
                writeSSEChunk(sseStream, requestId, actualModel, { reasoning_content: filtered })
              }
            }
            if (message?.content) {
              const filtered = filterInternalMarkers(message.content)
              if (filtered) {
                allCollectedContent += filtered
                writeSSEChunk(sseStream, requestId, actualModel, { content: filtered })
              }
            }
            if (message?.tool_calls && message.tool_calls.length > 0) {
              for (const tc of message.tool_calls) {
                writeSSEChunk(sseStream, requestId, actualModel, {
                  tool_calls: [tc]
                })
              }
            }
          }

          // 处理 Agent 会话状态
          if (agentSessionId && body) {
            const hasToolCalls = agentLoopManager.processModelResponse(agentSessionId, body)
            if (!hasToolCalls) {
              // 循环结束
              writeSSEChunk(sseStream, requestId, actualModel, {}, 'stop')
              sseStream.write('data: [DONE]\n\n')
              sseStream.end()
              break
            }

            // 有 tool_calls，需要继续循环
            const session = agentLoopManager.getSession(agentSessionId)
            if (session) {
              const nextRequest = agentLoopManager.buildNextStreamRequest(session, originalRequest)
              request.messages = nextRequest.messages
              if (nextRequest.tools) request.tools = nextRequest.tools
              if (nextRequest.tool_choice !== undefined) request.tool_choice = nextRequest.tool_choice
            }
          }
          continue
        }

        // 流式结果：实时处理并转发
        try {
          const streamResult = await processStreamInRealTime(
            result.stream,
            sseStream,
            requestId,
            actualModel,
            result.skipTransform || false
          )

          allCollectedContent += streamResult.content
          allCollectedReasoning += streamResult.reasoningContent

          // 处理 Agent 会话状态
          if (agentSessionId) {
            const hasToolCalls = agentLoopManager.processStreamResponse(
              agentSessionId,
              streamResult.content,
              streamResult.toolCalls,
              streamResult.reasoningContent
            )

            if (!hasToolCalls) {
              // 循环结束
              writeSSEChunk(sseStream, requestId, actualModel, {}, 'stop')
              sseStream.write('data: [DONE]\n\n')
              sseStream.end()
              break
            }

            // 有 tool_calls，需要继续循环
            // 写入工具调用信息提示
            const toolNames = streamResult.toolCalls.map(tc => tc.function?.name).join(', ')
            writeAgentStepSeparator(sseStream, requestId, actualModel, `Tool calls: ${toolNames}`)

            const session = agentLoopManager.getSession(agentSessionId)
            if (session) {
              const nextRequest = agentLoopManager.buildNextStreamRequest(session, originalRequest)
              request.messages = nextRequest.messages
              if (nextRequest.tools) request.tools = nextRequest.tools
              if (nextRequest.tool_choice !== undefined) request.tool_choice = nextRequest.tool_choice
            }
          } else {
            // 无会话，直接结束
            writeSSEChunk(sseStream, requestId, actualModel, {}, 'stop')
            sseStream.write('data: [DONE]\n\n')
            sseStream.end()
            break
          }
        } catch (streamError) {
          console.error('[Chat] Agent stream processing error:', streamError)
          const errorMsg = streamError instanceof Error ? streamError.message : 'Stream processing error'
          writeSSEChunk(sseStream, requestId, actualModel, {
            content: `\n\n❌ **Stream error in round ${currentRound}:** ${errorMsg}\n`
          })
          writeSSEChunk(sseStream, requestId, actualModel, {}, 'stop')
          sseStream.write('data: [DONE]\n\n')
          sseStream.end()
          break
        }

        // 检查是否超过最大轮次
        if (currentRound >= maxRounds) {
          writeSSEChunk(sseStream, requestId, actualModel, {
            content: `\n\n⚠️ **Agent loop reached maximum rounds (${maxRounds})**\n`
          })
          writeSSEChunk(sseStream, requestId, actualModel, {}, 'stop')
          sseStream.write('data: [DONE]\n\n')
          sseStream.end()
          if (agentSessionId) {
            agentLoopManager.deleteSession(agentSessionId)
          }
          break
        }
      }

      // 记录日志
      const latency = Date.now() - startTime
      const userInput = extractUserInput(request.messages)
      storeManager.addRequestLog({
        timestamp: startTime,
        status: 'success',
        statusCode: 200,
        method: 'POST',
        url: '/v1/chat/completions',
        model: request.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(originalRequest),
        userInput,
        webSearch: request.web_search,
        reasoningEffort: request.reasoning_effort,
        responseStatus: 200,
        responseBody: allCollectedContent || undefined,
        latency,
        isStream: true,
      })

      storeManager.recordRequestInStats(true, latency, request.model, provider.id, account.id)
      proxyStatusManager.recordRequestSuccess(latency)

      storeManager.updateAccount(account.id, {
        lastUsed: Date.now(),
        requestCount: (account.requestCount || 0) + 1,
        todayUsed: (account.todayUsed || 0) + 1,
      })

      return
    }

    // ============================================================
    // 非 Agent 模式 或 Agent 模式 + 客户端非流式
    // ============================================================
    const result = await requestForwarder.forwardChatCompletion(
      request,
      account,
      provider,
      actualModel,
      context
    )

    const latency = Date.now() - startTime

    if (!result.success) {
      proxyStatusManager.recordRequestFailure(latency)

      if (result.status && result.status >= 400 && result.status !== 429) {
        loadBalancer.markAccountFailed(account.id)
      }

      ctx.status = result.status || 500
      ctx.body = {
        error: {
          message: result.error || 'Request failed',
          type: 'api_error',
          param: null,
          code: null,
        },
      }

      storeManager.addLog('error', `Request failed: ${result.error}`, {
        requestId,
        providerId: provider.id,
        accountId: account.id,
        model: request.model,
        latency,
      })

      const userInput = extractUserInput(request.messages)
      const errorResponseBody = JSON.stringify({
        error: {
          message: result.error || 'Request failed',
          type: 'api_error',
          param: null,
          code: null,
        },
      })
      storeManager.addRequestLog({
        timestamp: startTime,
        status: 'error',
        statusCode: result.status || 500,
        method: 'POST',
        url: '/v1/chat/completions',
        model: request.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(request),
        userInput,
        webSearch: request.web_search,
        reasoningEffort: request.reasoning_effort,
        responseStatus: result.status || 500,
        responseBody: errorResponseBody,
        latency,
        isStream: request.stream || false,
        errorMessage: result.error,
      })

      storeManager.recordRequestInStats(false, latency, request.model, provider.id, account.id)

      return
    }

    loadBalancer.clearAccountFailure(account.id)

    proxyStatusManager.recordRequestSuccess(latency)

    storeManager.updateAccount(account.id, {
      lastUsed: Date.now(),
      requestCount: (account.requestCount || 0) + 1,
      todayUsed: (account.todayUsed || 0) + 1,
    })

    storeManager.addLog('debug', `Request succeeded`, {
      requestId,
      providerId: provider.id,
      accountId: account.id,
      model: request.model,
      actualModel,
      latency,
      isStream: request.stream,
    })

    const userInput = extractUserInput(request.messages)
    // Prepare response body for logging (only for non-stream requests)
    const responseBodyForLog = !request.stream && result.body
      ? JSON.stringify(result.body)
      : undefined

    // For streaming requests, we'll collect content and update the log later
    let logEntryId: string | undefined

    if (!request.stream) {
      // Non-streaming: record log with response body now
      const logEntry = storeManager.addRequestLog({
        timestamp: startTime,
        status: 'success',
        statusCode: 200,
        method: 'POST',
        url: '/v1/chat/completions',
        model: request.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(request),
        userInput,
        webSearch: request.web_search,
        reasoningEffort: request.reasoning_effort,
        responseStatus: 200,
        responseBody: responseBodyForLog,
        latency,
        isStream: false,
      })
      logEntryId = logEntry.id
    } else {
      // Streaming: record log now, will update response body later
      const logEntry = storeManager.addRequestLog({
        timestamp: startTime,
        status: 'success',
        statusCode: 200,
        method: 'POST',
        url: '/v1/chat/completions',
        model: request.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(request),
        userInput,
        webSearch: request.web_search,
        reasoningEffort: request.reasoning_effort,
        responseStatus: 200,
        latency,
        isStream: true,
      })
      logEntryId = logEntry.id
    }

    storeManager.recordRequestInStats(true, latency, request.model, provider.id, account.id)

    if (request.stream && result.stream) {
      // 客户端要求流式，且后端返回了流 → 正常流式响应
      ctx.set('Content-Type', 'text/event-stream')
      ctx.set('Cache-Control', 'no-cache')
      ctx.set('Connection', 'keep-alive')
      ctx.set('X-Accel-Buffering', 'no')

      // Create a wrapper stream to handle errors and collect content
      const wrapperStream = new PassThrough()

      // Collect stream content for logging (raw SSE output)
      let collectedContent = ''

      // Handle stream errors
      result.stream.once('error', (err: Error) => {
        console.error('[Chat] Stream error:', err.message)

        // Send error as SSE event
        const errorEvent = {
          id: requestId,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model: actualModel,
          choices: [{
            index: 0,
            delta: {
              content: `\n\n[Error: ${err.message}]`,
            },
            finish_reason: 'stop',
          }],
        }

        wrapperStream.write(`data: ${JSON.stringify(errorEvent)}\n\n`)
        wrapperStream.write('data: [DONE]\n\n')
        wrapperStream.end()

        storeManager.addLog('error', `Stream error: ${err.message}`, {
          requestId,
          providerId: provider.id,
          accountId: account.id,
          model: request.model,
        })
      })

      // Check if stream is already in correct SSE format (from adapters like Kimi, GLM, DeepSeek)
      if (result.skipTransform) {
        // Stream is already formatted, pipe through wrapper and collect
        // v1.7.2: 过滤内部标记后实时推送
        let streamBuffer = ''
        result.stream.on('data', (chunk: Buffer) => {
          const text = chunk.toString()
          streamBuffer += text

          // 按行处理，过滤内部标记
          const lines = streamBuffer.split('\n')
          streamBuffer = lines.pop() || ''

          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed || trimmed.startsWith(':')) {
              wrapperStream.write(line + '\n')
              continue
            }

            if (trimmed.startsWith('data: ')) {
              const data = trimmed.slice(6)
              if (data === '[DONE]') {
                collectedContent += text
                wrapperStream.write(line + '\n')
                continue
              }

              try {
                const parsed = JSON.parse(data)
                // 过滤 content 中的内部标记
                if (parsed.choices?.[0]?.delta?.content) {
                  const filtered = filterInternalMarkers(parsed.choices[0].delta.content)
                  if (filtered !== parsed.choices[0].delta.content) {
                    parsed.choices[0].delta.content = filtered
                  }
                }
                // 过滤 reasoning_content 中的内部标记
                if (parsed.choices?.[0]?.delta?.reasoning_content) {
                  const filtered = filterInternalMarkers(parsed.choices[0].delta.reasoning_content)
                  if (filtered !== parsed.choices[0].delta.reasoning_content) {
                    parsed.choices[0].delta.reasoning_content = filtered
                  }
                }
                collectedContent += text
                wrapperStream.write(`data: ${JSON.stringify(parsed)}\n`)
              } catch {
                collectedContent += text
                wrapperStream.write(line + '\n')
              }
            } else {
              wrapperStream.write(line + '\n')
            }
          }
        })

        // When source stream ends normally, update log and end wrapper
        result.stream.once('end', () => {
          // 处理剩余 buffer
          if (streamBuffer.trim()) {
            wrapperStream.write(streamBuffer)
            collectedContent += streamBuffer
          }

          // Update log with collected response
          if (logEntryId) {
            storeManager.updateRequestLog(logEntryId, {
              responseBody: collectedContent || undefined,
            })
          }
          wrapperStream.end()
        })
      } else {
        // Need to transform the stream
        const transformStream = streamHandler.createTransformStream(
          actualModel,
          requestId,
          () => {
            storeManager.addLog('debug', `Stream response completed`, { requestId })
          }
        )

        // Collect from transform stream output
        transformStream.on('data', (chunk: Buffer) => {
          collectedContent += chunk.toString()
        })

        result.stream.pipe(transformStream)
        transformStream.pipe(wrapperStream, { end: false })

        transformStream.once('end', () => {
          // Update log with collected response
          if (logEntryId) {
            storeManager.updateRequestLog(logEntryId, {
              responseBody: collectedContent || undefined,
            })
          }
          wrapperStream.end()
        })
      }

      ctx.body = wrapperStream
    } else {
      // ============================================================
      // 非流式响应（或 Agent 模式下非流式请求的结果）
      // ============================================================

      // Agent 模式：客户端要求流式，但内部用了非流式请求 → 需要把 JSON 转为 SSE 格式
      if (agentModeEnabled && clientWantsStream && result.body) {
        // 先处理 Agent 会话状态
        if (agentSessionId) {
          const hasToolCalls = agentLoopManager.processModelResponse(agentSessionId, result.body)
          if (hasToolCalls) {
            ctx.set('X-Session-Id', agentSessionId)
            ctx.set('X-Agent-Loop', 'waiting_tool_result')
            ctx.set('X-Agent-Round', String(agentLoopManager.getRoundCount(agentSessionId)))
          } else {
            ctx.set('X-Session-Id', agentSessionId)
            ctx.set('X-Agent-Loop', 'completed')
            ctx.set('X-Agent-Round', String(agentLoopManager.getRoundCount(agentSessionId)))
          }
        }

        // 将 JSON 响应转换为 SSE 流式格式返回给客户端
        ctx.set('Content-Type', 'text/event-stream')
        ctx.set('Cache-Control', 'no-cache')
        ctx.set('Connection', 'keep-alive')
        ctx.set('X-Accel-Buffering', 'no')

        const responseBody = isAnthropicToolFormat(request.tool_format)
          ? transformResponseToAnthropic(result.body)
          : result.body

        const sseStream = streamHandler.createPassThrough()

        // 将完整响应拆分为 SSE chunk 格式
        const message = responseBody?.choices?.[0]?.message
        if (message) {
          // 第一个 chunk: role
          sseStream.write(`data: ${JSON.stringify({
            id: responseBody?.id || requestId,
            object: 'chat.completion.chunk',
            created: responseBody?.created || Math.floor(Date.now() / 1000),
            model: responseBody?.model || actualModel,
            choices: [{
              index: 0,
              delta: { role: 'assistant' },
              finish_reason: null,
            }],
          })}\n\n`)

          // 第二个 chunk: reasoning_content（思考内容，如果有）
          if (message.reasoning_content) {
            const filtered = filterInternalMarkers(message.reasoning_content)
            if (filtered) {
              sseStream.write(`data: ${JSON.stringify({
                id: responseBody?.id || requestId,
                object: 'chat.completion.chunk',
                created: responseBody?.created || Math.floor(Date.now() / 1000),
                model: responseBody?.model || actualModel,
                choices: [{
                  index: 0,
                  delta: { reasoning_content: filtered },
                  finish_reason: null,
                }],
              })}\n\n`)
            }
          }

          // 第三个 chunk: content（正文内容）
          if (message.content) {
            const filtered = filterInternalMarkers(message.content)
            if (filtered) {
              sseStream.write(`data: ${JSON.stringify({
                id: responseBody?.id || requestId,
                object: 'chat.completion.chunk',
                created: responseBody?.created || Math.floor(Date.now() / 1000),
                model: responseBody?.model || actualModel,
                choices: [{
                  index: 0,
                  delta: { content: filtered },
                  finish_reason: null,
                }],
              })}\n\n`)
            }
          }

          // 第四个 chunk: tool_calls（如果有）
          if (message.tool_calls && message.tool_calls.length > 0) {
            for (const tc of message.tool_calls) {
              sseStream.write(`data: ${JSON.stringify({
                id: responseBody?.id || requestId,
                object: 'chat.completion.chunk',
                created: responseBody?.created || Math.floor(Date.now() / 1000),
                model: responseBody?.model || actualModel,
                choices: [{
                  index: 0,
                  delta: {
                    tool_calls: [{
                      index: tc.index || 0,
                      id: tc.id,
                      type: 'function',
                      function: { name: tc.function.name, arguments: tc.function.arguments },
                    }],
                  },
                  finish_reason: null,
                }],
              })}\n\n`)
            }
          }

          // 最后一个 chunk: finish_reason
          const finishReason = message.tool_calls?.length > 0 ? 'tool_calls' : 'stop'
          sseStream.write(`data: ${JSON.stringify({
            id: responseBody?.id || requestId,
            object: 'chat.completion.chunk',
            created: responseBody?.created || Math.floor(Date.now() / 1000),
            model: responseBody?.model || actualModel,
            choices: [{
              index: 0,
              delta: {},
              finish_reason: finishReason,
            }],
          })}\n\n`)
        }

        // [DONE] 标记
        sseStream.write('data: [DONE]\n\n')
        sseStream.end()
        ctx.body = sseStream

        // 记录日志
        if (logEntryId) {
          storeManager.updateRequestLog(logEntryId, {
            responseBody: JSON.stringify(responseBody),
          })
        }

        console.log(`[Chat] Agent: converted non-stream response to SSE for streaming client`)
      } else {
        // 标准非流式响应
        ctx.set('Content-Type', 'application/json')

        if (result.body) {
          // Check if we need to transform to Anthropic format
          if (isAnthropicToolFormat(request.tool_format)) {
            ctx.body = transformResponseToAnthropic(result.body)
            console.log('[Chat] Transformed response to Anthropic tool format')
          } else {
            ctx.body = result.body
          }

          // ============================================================
          // Agent Loop: 检测模型响应中的 tool_calls，更新会话状态
          // ============================================================
          if (agentModeEnabled && agentSessionId && result.body) {
            const hasToolCalls = agentLoopManager.processModelResponse(agentSessionId, result.body)

            if (hasToolCalls) {
              // 响应包含 tool_calls → 会话保持活跃，等待前端回传工具执行结果
              ctx.set('X-Session-Id', agentSessionId)
              ctx.set('X-Agent-Loop', 'waiting_tool_result')
              ctx.set('X-Agent-Round', String(agentLoopManager.getRoundCount(agentSessionId)))
              console.log(`[Chat] Agent: tool_calls detected, waiting for tool results. Session: ${agentSessionId}, Round: ${agentLoopManager.getRoundCount(agentSessionId)}`)
            } else {
              // 模型输出最终文本回复 → 循环结束，会话已自动清理
              ctx.set('X-Session-Id', agentSessionId || '')
              ctx.set('X-Agent-Loop', 'completed')
              ctx.set('X-Agent-Round', String(agentLoopManager.getRoundCount(agentSessionId)))
              console.log(`[Chat] Agent: final response received, loop completed. Session: ${agentSessionId}, Total rounds: ${agentLoopManager.getRoundCount(agentSessionId)}`)
            }
          }
        } else {
          ctx.body = {
            id: requestId,
            object: 'chat.completion',
            created: Math.floor(Date.now() / 1000),
            model: actualModel,
            choices: [{
              index: 0,
              message: {
                role: 'assistant',
                content: '',
              },
              finish_reason: 'stop',
            }],
            usage: {
              prompt_tokens: 0,
              completion_tokens: 0,
              total_tokens: 0,
            },
          }
        }
      }
    }
  } catch (error) {
    const latency = Date.now() - startTime
    proxyStatusManager.recordRequestFailure(latency)

    const errorMessage = error instanceof Error ? error.message : 'Unknown error'
    const errorStack = error instanceof Error ? error.stack : undefined

    ctx.status = 500
    ctx.body = {
      error: {
        message: errorMessage,
        type: 'internal_error',
        param: null,
        code: null,
      },
    }

    storeManager.addLog('error', `Request exception: ${errorMessage}`, {
      requestId,
      providerId: provider.id,
      accountId: account.id,
      model: request.model,
      latency,
      error: errorMessage,
    })

    const userInput = extractUserInput(request.messages)
    const exceptionResponseBody = JSON.stringify({
      error: {
        message: errorMessage,
        type: 'internal_error',
        param: null,
        code: null,
      },
    })
    storeManager.addRequestLog({
      timestamp: startTime,
      status: 'error',
      statusCode: 500,
      method: 'POST',
      url: '/v1/chat/completions',
      model: request.model,
      actualModel,
      providerId: provider.id,
      providerName: provider.name,
      accountId: account.id,
      accountName: account.name,
      requestBody: JSON.stringify(request),
      userInput,
      webSearch: request.web_search,
      reasoningEffort: request.reasoning_effort,
      responseStatus: 500,
      responseBody: exceptionResponseBody,
      latency,
      isStream: request.stream || false,
      errorMessage,
      errorStack,
    })

    storeManager.recordRequestInStats(false, latency, request.model, provider.id, account.id)
  }
})

export default router
