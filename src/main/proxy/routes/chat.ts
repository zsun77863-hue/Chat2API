/**
 * Proxy Service Module - Chat Completions Route
 * Implements /v1/chat/completions route
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
import { agentLoopManager } from '../agentLoop'

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
  // 只要请求体包含 tools 定义，即自动激活 Agent 循环
  // 可通过请求头 X-Agent-Mode: false 手动关闭
  // 激活后：缓存会话上下文 → 检测 tool_calls → 自动推进多轮循环
  // ============================================================
  const agentModeEnabled = agentLoopManager.shouldEnable(request, ctx.headers as any)
  let agentSessionId: string | undefined
  // 保存客户端原始的 stream 设置，用于决定响应格式
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
        const nextRequest = agentLoopManager.buildNextRequest(session, request)
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

    // Agent 循环内部强制非流式请求，以便完整解析 tool_calls
    // 但不修改 request.stream，因为后续响应格式仍按客户端原始要求返回
    request.stream = false
  }

  try {
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

    if ((clientWantsStream || request.stream === true) && result.stream) {
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
        result.stream.on('data', (chunk: Buffer) => {
          collectedContent += chunk.toString()
        })

        result.stream.pipe(wrapperStream, { end: false })

        // When source stream ends normally, update log and end wrapper
        result.stream.once('end', () => {
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
      // 非流式响应（或 Agent 模式下强制非流式请求的结果）
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
            sseStream.write(`data: ${JSON.stringify({
              id: responseBody?.id || requestId,
              object: 'chat.completion.chunk',
              created: responseBody?.created || Math.floor(Date.now() / 1000),
              model: responseBody?.model || actualModel,
              choices: [{
                index: 0,
                delta: { reasoning_content: message.reasoning_content },
                finish_reason: null,
              }],
            })}\n\n`)
          }

          // 第三个 chunk: content（正文内容）
          if (message.content) {
            sseStream.write(`data: ${JSON.stringify({
              id: responseBody?.id || requestId,
              object: 'chat.completion.chunk',
              created: responseBody?.created || Math.floor(Date.now() / 1000),
              model: responseBody?.model || actualModel,
              choices: [{
                index: 0,
                delta: { content: message.content },
                finish_reason: null,
              }],
            })}\n\n`)
          }

          // 第三个 chunk: tool_calls（如果有）
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
