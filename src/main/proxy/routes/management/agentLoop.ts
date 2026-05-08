/**
 * Management Route - Agent Loop (v1.4.1)
 * Provides status, control, and diagnostics for the agent loop system
 */

import Router from '@koa/router'
import type { Context } from 'koa'
import { managementAuthMiddleware } from '../../middleware/managementAuth'
import { agentLoopManager } from '../../agentLoop'
import type { ManagementApiResponse } from '../../../../shared/types'

const router = new Router({ prefix: '/v0/management/agent-loop' })

router.use(managementAuthMiddleware)

/**
 * Get agent loop status and stats
 */
router.get('/status', async (ctx: Context) => {
  const stats = agentLoopManager.getStats()
  const activeSessions = agentLoopManager.getActiveSessions()

  ctx.body = {
    success: true,
    data: {
      stats,
      activeSessions: activeSessions.map(({ sessionId, session }) => ({
        sessionId,
        model: session.model,
        roundNumber: session.roundNumber,
        messageCount: session.messages.length,
        hasTools: !!session.tools && session.tools.length > 0,
        createdAt: session.createdAt,
        lastActiveAt: session.lastActiveAt,
        clientIP: session.clientIP,
        anomalyCount: session.anomalyCount,
      })),
    },
  } as ManagementApiResponse
})

/**
 * Get details of a specific session
 */
router.get('/sessions/:sessionId', async (ctx: Context) => {
  const { sessionId } = ctx.params
  const session = agentLoopManager.getSession(sessionId)

  if (!session) {
    ctx.status = 404
    ctx.body = {
      success: false,
      error: 'Session not found',
    } as ManagementApiResponse
    return
  }

  ctx.body = {
    success: true,
    data: {
      sessionId,
      model: session.model,
      actualModel: session.actualModel,
      providerId: session.providerId,
      accountId: session.accountId,
      roundNumber: session.roundNumber,
      completed: session.completed,
      messageCount: session.messages.length,
      messages: session.messages.map((msg, i) => ({
        index: i,
        role: msg.role,
        hasToolCalls: !!(msg.tool_calls && msg.tool_calls.length > 0),
        hasToolCallId: !!msg.tool_call_id,
        contentPreview: typeof msg.content === 'string'
          ? msg.content.substring(0, 200)
          : msg.content ? '[complex]' : null,
      })),
      tools: session.tools?.map(t => t.function?.name) || [],
      createdAt: session.createdAt,
      lastActiveAt: session.lastActiveAt,
      clientIP: session.clientIP,
      anomalyCount: session.anomalyCount,
      // Full generation params for debugging
      params: {
        temperature: session.temperature,
        top_p: session.top_p,
        max_tokens: session.max_tokens,
        tool_choice: session.tool_choice,
        tool_format: session.tool_format,
      },
    },
  } as ManagementApiResponse
})

/**
 * Delete a specific session
 */
router.delete('/sessions/:sessionId', async (ctx: Context) => {
  const { sessionId } = ctx.params
  const deleted = agentLoopManager.deleteSession(sessionId)

  ctx.body = {
    success: deleted,
    data: { sessionId, deleted },
  } as ManagementApiResponse
})

/**
 * Delete all sessions for a specific client IP
 */
router.delete('/clients/:clientIP', async (ctx: Context) => {
  const { clientIP } = ctx.params
  const count = agentLoopManager.deleteClientSessions(decodeURIComponent(clientIP))

  ctx.body = {
    success: true,
    data: { clientIP, deletedCount: count },
  } as ManagementApiResponse
})

/**
 * Force-reset a specific session
 */
router.post('/sessions/:sessionId/reset', async (ctx: Context) => {
  const { sessionId } = ctx.params
  const reset = agentLoopManager.resetSession(sessionId)

  ctx.body = {
    success: reset,
    data: { sessionId, reset },
  } as ManagementApiResponse
})

/**
 * Get agent loop configuration info
 */
router.get('/config', async (ctx: Context) => {
  ctx.body = {
    success: true,
    data: {
      version: '1.4.1',
      maxRounds: 15,
      sessionTtlMs: 30 * 60 * 1000,
      cleanupIntervalMs: 60 * 1000,
      maxSessionsPerClient: 10,
      maxTotalSessions: 500,
      maxAnomalyCount: 3,
      description: 'Agent loop automatically handles multi-turn tool-calling sessions. Enable via X-Agent-Loop: true header. v1.4.1 adds structured tool_calls detection, anomaly circuit-breaker, per-client isolation, graceful degradation, and full param inheritance.',
    },
  } as ManagementApiResponse
})

export default router
