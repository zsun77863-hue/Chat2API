/**
 * Claude Messages API Route Handler
 * Implements the Anthropic Messages API (/v1/messages) endpoint
 * Converts Claude format requests to OpenAI format, processes them through
 * the existing chat2api pipeline, and converts responses back to Claude format.
 */

import Router from '@koa/router'
import type { Context } from 'koa'
import { PassThrough } from 'stream'
import { ClaudeMessageRequest, ClaudeErrorResponse } from './types'
import {
  claudeRequestToOpenAI,
  openaiResponseToClaude,
  ClaudeStreamConverter,
  generateClaudeMessageId,
} from './converter'
import { ChatCompletionRequest, ProxyContext } from '../types'
import { loadBalancer } from '../loadbalancer'
import { requestForwarder } from '../forwarder'
import { streamHandler } from '../stream'
import { proxyStatusManager } from '../status'
import { modelMapper } from '../modelMapper'
import { storeManager } from '../../store/store'

const router = new Router({ prefix: '/v1' })

/**
 * Get Client IP
 */
function getClientIP(ctx: Context): string {
  return (ctx.headers['x-real-ip'] as string) ||
    (ctx.headers['x-forwarded-for'] as string) ||
    ctx.ip ||
    'unknown'
}

/**
 * Extract user input from Claude messages (last user message, full content)
 */
function extractUserInputFromClaude(messages: Array<{ role: string; content?: string | any[] | null }>): string | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role === 'user' && msg.content) {
      if (typeof msg.content === 'string') {
        return msg.content
      } else if (Array.isArray(msg.content)) {
        const textParts = msg.content.filter((p: any) => p.type === 'text')
        if (textParts.length > 0) {
          return textParts.map((p: any) => p.text || '').join(' ')
        }
      }
    }
  }
  return undefined
}

/**
 * Send a Claude-formatted error response
 */
function sendClaudeError(ctx: Context, status: number, errorType: string, message: string): void {
  ctx.status = status
  ctx.set('Content-Type', 'application/json')
  ctx.set('X-Request-ID', generateClaudeMessageId())
  const errorResponse: ClaudeErrorResponse = {
    type: 'error',
    error: {
      type: errorType as any,
      message,
    },
  }
  ctx.body = errorResponse
}

/**
 * Handle Claude Messages API Request
 * POST /v1/messages
 */
router.post('/messages', async (ctx: Context) => {
  const startTime = Date.now()
  const requestId = generateClaudeMessageId()
  const clientIP = getClientIP(ctx)

  let claudeReq: ClaudeMessageRequest
  try {
    claudeReq = ctx.request.body as ClaudeMessageRequest
  } catch (error) {
    sendClaudeError(ctx, 400, 'invalid_request_error', 'Invalid request body')
    return
  }

  // Validate required fields
  if (!claudeReq.model) {
    sendClaudeError(ctx, 400, 'invalid_request_error', 'model is required')
    return
  }

  if (!claudeReq.messages || !Array.isArray(claudeReq.messages) || claudeReq.messages.length === 0) {
    sendClaudeError(ctx, 400, 'invalid_request_error', 'messages: at least one message is required')
    return
  }

  if (!claudeReq.max_tokens) {
    sendClaudeError(ctx, 400, 'invalid_request_error', 'max_tokens is required')
    return
  }

  const isStream = claudeReq.stream === true

  console.log(`[Claude] Request: model=${claudeReq.model}, stream=${isStream}, messages=${claudeReq.messages.length}, tools=${claudeReq.tools?.length || 0}`)

  try {
    // Convert Claude request to OpenAI format
    const openaiReq = claudeRequestToOpenAI(claudeReq)

    // Select account through load balancer
    const config = storeManager.getConfig()
    const preferredProviderId = modelMapper.getPreferredProvider(openaiReq.model)
    const preferredAccountId = modelMapper.getPreferredAccount(openaiReq.model)

    const selection = loadBalancer.selectAccount(
      openaiReq.model,
      config.loadBalanceStrategy,
      preferredProviderId,
      preferredAccountId
    )

    if (!selection) {
      sendClaudeError(ctx, 503, 'api_error', `No available account for model: ${claudeReq.model}`)
      return
    }

    const { account, provider, actualModel } = selection

    const context: ProxyContext = {
      requestId,
      providerId: provider.id,
      accountId: account.id,
      model: openaiReq.model,
      actualModel,
      startTime,
      isStream,
      clientIP,
    }

    proxyStatusManager.recordRequestStart(openaiReq.model, provider.id, account.id)

    // Forward through the existing pipeline
    const result = await requestForwarder.forwardChatCompletion(
      openaiReq,
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

      // Map HTTP status to Claude error types
      let errorType = 'api_error'
      if (result.status === 400) errorType = 'invalid_request_error'
      else if (result.status === 401) errorType = 'authentication_error'
      else if (result.status === 403) errorType = 'permission_error'
      else if (result.status === 404) errorType = 'not_found_error'
      else if (result.status === 429) errorType = 'rate_limit_error'
      else if (result.status === 500) errorType = 'api_error'
      else if (result.status === 529) errorType = 'overloaded_error'

      sendClaudeError(ctx, result.status || 500, errorType, result.error || 'Request failed')

      storeManager.addLog('error', `[Claude] Request failed: ${result.error}`, {
        requestId,
        providerId: provider.id,
        accountId: account.id,
        model: claudeReq.model,
        latency,
      })

      const userInput = extractUserInputFromClaude(claudeReq.messages)
      storeManager.addRequestLog({
        timestamp: startTime,
        status: 'error',
        statusCode: result.status || 500,
        method: 'POST',
        url: '/v1/messages',
        model: claudeReq.model,
        actualModel,
        providerId: provider.id,
        providerName: provider.name,
        accountId: account.id,
        accountName: account.name,
        requestBody: JSON.stringify(claudeReq),
        userInput,
        responseStatus: result.status || 500,
        responseBody: JSON.stringify({ type: 'error', error: { type: errorType, message: result.error } }),
        latency,
        isStream,
        errorMessage: result.error,
      })

      storeManager.recordRequestInStats(false, latency, openaiReq.model, provider.id, account.id)
      return
    }

    loadBalancer.clearAccountFailure(account.id)
    proxyStatusManager.recordRequestSuccess(latency)

    storeManager.updateAccount(account.id, {
      lastUsed: Date.now(),
      requestCount: (account.requestCount || 0) + 1,
      todayUsed: (account.todayUsed || 0) + 1,
    })

    const userInput = extractUserInputFromClaude(claudeReq.messages)

    if (isStream && result.stream) {
      // ============================================================
      // Streaming Response: Convert OpenAI SSE to Claude SSE
      // ============================================================
      ctx.set('Content-Type', 'text/event-stream')
      ctx.set('Cache-Control', 'no-cache')
      ctx.set('Connection', 'keep-alive')
      ctx.set('X-Accel-Buffering', 'no')
      ctx.set('X-Request-ID', requestId)
      ctx.set('Anthropic-Ratelimit-Requests-Limit', '1000')
      ctx.set('Anthropic-Ratelimit-Requests-Remaining', '999')

      const wrapperStream = new PassThrough()
      const converter = new ClaudeStreamConverter(requestId, claudeReq.model)

      // Collect stream content for logging
      let collectedContent = ''

      // Handle stream errors
      result.stream.once('error', (err: Error) => {
        console.error('[Claude] Stream error:', err.message)

        const errorEvent = {
          type: 'error' as const,
          error: {
            type: 'api_error',
            message: err.message,
          },
        }
        wrapperStream.write(`event: error\ndata: ${JSON.stringify(errorEvent)}\n\n`)
        wrapperStream.end()

        storeManager.addLog('error', `[Claude] Stream error: ${err.message}`, {
          requestId,
          providerId: provider.id,
          accountId: account.id,
          model: claudeReq.model,
        })
      })

      // Process the OpenAI stream and convert to Claude format
      const processStream = async () => {
        try {
          if (result.skipTransform) {
            // Stream is already in SSE format, need to parse and convert
            for await (const chunk of result.stream as AsyncIterable<Buffer>) {
              const chunkStr = chunk.toString()
              collectedContent += chunkStr

              const lines = chunkStr.split('\n')
              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const data = line.slice(6).trim()
                  if (data === '[DONE]') {
                    const finalEvents = converter.finalize()
                    for (const event of finalEvents) {
                      wrapperStream.write(event)
                    }
                    continue
                  }
                  try {
                    const parsed = JSON.parse(data)
                    const claudeEvents = converter.convertChunk(parsed)
                    for (const event of claudeEvents) {
                      wrapperStream.write(event)
                    }
                  } catch {
                    // Skip unparseable chunks
                  }
                }
              }
            }
          } else {
            // Stream needs transformation through streamHandler first
            const transformStream = streamHandler.createTransformStream(
              actualModel,
              requestId,
              () => {
                storeManager.addLog('debug', `[Claude] Stream response completed`, { requestId })
              }
            )

            // We need to collect the transformed output and then convert it
            const transformOutput = new PassThrough()
            transformStream.pipe(transformOutput)

            result.stream!.pipe(transformStream)

            for await (const chunk of transformOutput as AsyncIterable<Buffer>) {
              const chunkStr = chunk.toString()
              collectedContent += chunkStr

              const lines = chunkStr.split('\n')
              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  const data = line.slice(6).trim()
                  if (data === '[DONE]') {
                    const finalEvents = converter.finalize()
                    for (const event of finalEvents) {
                      wrapperStream.write(event)
                    }
                    continue
                  }
                  try {
                    const parsed = JSON.parse(data)
                    const claudeEvents = converter.convertChunk(parsed)
                    for (const event of claudeEvents) {
                      wrapperStream.write(event)
                    }
                  } catch {
                    // Skip unparseable chunks
                  }
                }
              }
            }
          }

          // Send ping
          wrapperStream.write(`event: ping\ndata: ${JSON.stringify({ type: 'ping' })}\n\n`)
          wrapperStream.end()

          // Update log
          storeManager.addRequestLog({
            timestamp: startTime,
            status: 'success',
            statusCode: 200,
            method: 'POST',
            url: '/v1/messages',
            model: claudeReq.model,
            actualModel,
            providerId: provider.id,
            providerName: provider.name,
            accountId: account.id,
            accountName: account.name,
            requestBody: JSON.stringify(claudeReq),
            userInput,
            responseStatus: 200,
            responseBody: collectedContent || undefined,
            latency,
            isStream: true,
          })

          console.log(`[Claude] Streaming response completed: ${requestId}`)
        } catch (streamError: any) {
          console.error(`[Claude] Stream processing error: ${streamError.message}`)
          try {
            wrapperStream.write(`event: error\ndata: ${JSON.stringify({ type: 'error', error: { type: 'api_error', message: streamError.message || 'Stream processing error' } })}\n\n`)
          } catch {
            // Connection may already be closed
          }
          wrapperStream.end()
        }
      }

      // Start processing the stream in the background
      processStream().catch(err => {
        console.error('[Claude] Fatal stream error:', err)
        try { wrapperStream.end() } catch {}
      })

      ctx.body = wrapperStream
    } else {
      // ============================================================
      // Non-Streaming Response: Convert OpenAI JSON to Claude JSON
      // ============================================================
      ctx.set('Content-Type', 'application/json')
      ctx.set('X-Request-ID', requestId)
      ctx.set('Anthropic-Ratelimit-Requests-Limit', '1000')
      ctx.set('Anthropic-Ratelimit-Requests-Remaining', '999')

      if (result.body) {
        const claudeResponse = openaiResponseToClaude(result.body, requestId)

        console.log(`[Claude] Non-streaming response: stop_reason=${claudeResponse.stop_reason}, content_blocks=${claudeResponse.content.length}`)

        ctx.body = claudeResponse

        storeManager.addRequestLog({
          timestamp: startTime,
          status: 'success',
          statusCode: 200,
          method: 'POST',
          url: '/v1/messages',
          model: claudeReq.model,
          actualModel,
          providerId: provider.id,
          providerName: provider.name,
          accountId: account.id,
          accountName: account.name,
          requestBody: JSON.stringify(claudeReq),
          userInput,
          responseStatus: 200,
          responseBody: JSON.stringify(claudeResponse),
          latency,
          isStream: false,
        })
      } else {
        // Empty response
        const emptyResponse = {
          id: requestId,
          type: 'message' as const,
          role: 'assistant' as const,
          content: [{ type: 'text' as const, text: '' }],
          model: claudeReq.model,
          stop_reason: 'end_turn' as const,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        }
        ctx.body = emptyResponse
      }
    }

    storeManager.recordRequestInStats(true, latency, openaiReq.model, provider.id, account.id)
  } catch (error: any) {
    const latency = Date.now() - startTime
    proxyStatusManager.recordRequestFailure(latency)

    const errorMessage = error instanceof Error ? error.message : 'Unknown error'

    console.error(`[Claude] Request exception: ${errorMessage}`)
    console.error(`[Claude] Stack: ${error.stack}`)

    sendClaudeError(ctx, 500, 'api_error', errorMessage || 'Internal server error')

    storeManager.addLog('error', `[Claude] Request exception: ${errorMessage}`, {
      requestId,
      model: claudeReq.model,
      latency,
      error: errorMessage,
    })

    storeManager.recordRequestInStats(false, latency, claudeReq.model, '', '')
  }
})

export default router
