/**
 * Management Route - Agent Loop
 * Provides status and control for the agent loop system
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
 * Get agent loop configuration info
 */
router.get('/config', async (ctx: Context) => {
  ctx.body = {
    success: true,
    data: {
      maxRounds: 15,
      sessionTtlMs: 30 * 60 * 1000,
      cleanupIntervalMs: 60 * 1000,
      description: 'Agent loop automatically handles multi-turn tool-calling sessions. Enable via X-Agent-Loop: true header.',
    },
  } as ManagementApiResponse
})

export default router
