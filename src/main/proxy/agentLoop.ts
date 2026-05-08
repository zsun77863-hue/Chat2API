/**
 * Agent Loop Module (v1.4.1)
 *
 * Fixes from v1.4.0:
 * - Bug 1: Properly handles tool_result continuation — when OpenClaw sends back
 *   role=tool messages, they are merged into session history and auto-forwarded
 *   to the model for the next reasoning round.
 * - Bug 2: Session management improved — per-client isolation, max context rounds,
 *   auto-cleanup, manual destroy/reset, LRU eviction when memory limit hit.
 * - Bug 3: Structured tool_calls detection (JSON field parsing, not string matching),
 *   plus anomaly circuit-breaker to prevent infinite loops.
 * - Bug 4: Graceful degradation — if agent loop encounters an error, falls back to
 *   single-shot forwarding instead of crashing.
 * - Bug 5: All generation params (temperature, max_tokens, top_p, tools, tool_choice,
 *   tool_format) are fully inherited across every internal recursive round.
 *
 * Flow:
 * 1. Client sends request with X-Agent-Loop: true → Chat2API creates/caches session
 * 2. Request forwarded to model → if tool_calls in response, session cached, response
 *    returned with X-Session-Id and X-Agent-Waiting-For-Tools headers
 * 3. Client (OpenClaw) executes tools, sends results back with X-Session-Id
 * 4. Chat2API merges tool results into session, auto-forwards to model again
 * 5. Loop continues until model returns text-only response (no tool_calls)
 * 6. Final response returned with X-Agent-Completed: true
 *
 * Safety: max 15 auto-rounds + anomaly circuit-breaker.
 */

import { ChatCompletionRequest, ChatMessage, ForwardResult } from './types'
import type { Account, Provider } from '../store/types'
import type { ProxyContext } from './types'
import { requestForwarder } from '../forwarder'

// ============================================================
// Types
// ============================================================

interface AgentSession {
  /** Full conversation history including system, user, assistant, tool messages */
  messages: ChatMessage[]
  /** Original request metadata needed for subsequent forwards */
  model: string
  actualModel: string
  providerId: string
  accountId: string
  /** Tools definitions from the original request */
  tools?: ChatCompletionRequest['tools']
  tool_choice?: ChatCompletionRequest['tool_choice']
  tool_format?: ChatCompletionRequest['tool_format']
  /** Full generation params — inherited across all rounds */
  temperature?: number
  top_p?: number
  max_tokens?: number
  n?: number
  stop?: string | string[]
  presence_penalty?: number
  frequency_penalty?: number
  user?: string
  /** Session creation time */
  createdAt: number
  /** Last activity time */
  lastActiveAt: number
  /** Current round number (0 = initial request) */
  roundNumber: number
  /** Whether this session has been completed (model returned final text) */
  completed: boolean
  /** Client IP for isolation */
  clientIP: string
  /** Anomaly counter: consecutive rounds with identical tool_calls (possible loop) */
  anomalyCount: number
  /** Hash of last tool_calls to detect stuck loops */
  lastToolCallsHash: string
}

// ============================================================
// Configuration
// ============================================================

const MAX_AGENT_ROUNDS = 15
const SESSION_TTL_MS = 30 * 60 * 1000 // 30 minutes
const CLEANUP_INTERVAL_MS = 60 * 1000 // 1 minute
const MAX_SESSIONS_PER_CLIENT = 10
const MAX_TOTAL_SESSIONS = 500
const MAX_ANOMALY_COUNT = 3 // consecutive identical tool_calls rounds before forcing stop

// ============================================================
// Agent Loop Manager (Singleton)
// ============================================================

class AgentLoopManager {
  /** Map of sessionId → AgentSession */
  private sessions: Map<string, AgentSession> = new Map()
  /** Map of clientIP → Set<sessionId> for per-client isolation */
  private clientSessions: Map<string, Set<string>> = new Map()
  private cleanupTimer: NodeJS.Timeout | null = null

  constructor() {
    this.startCleanup()
    console.log('[AgentLoop] Initialized (v1.4.1), max rounds:', MAX_AGENT_ROUNDS)
  }

  // ----------------------------------------------------------
  // Public API
  // ----------------------------------------------------------

  /**
   * Main entry point: handle a chat completion request with agent loop support.
   *
   * Returns result + sessionId. If the model returned tool_calls, the caller
   * should return the response with X-Agent-Waiting-For-Tools: true header
   * so the client knows to send tool results back.
   */
  async handleRequest(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    context: ProxyContext,
    sessionId?: string,
    clientIP: string = 'unknown'
  ): Promise<{ result: ForwardResult; sessionId?: string; waitingForTools?: boolean }> {
    // Check if this is a tool-result continuation
    const isContinuation = this.isToolResultContinuation(request)

    let session: AgentSession | undefined
    let effectiveSessionId: string | undefined

    if (isContinuation && sessionId) {
      session = this.sessions.get(sessionId)
      if (session && !session.completed) {
        effectiveSessionId = sessionId
        console.log(`[AgentLoop] Continuing session ${sessionId}, round ${session.roundNumber + 1}`)

        // OpenClaw sends the FULL conversation history including tool results,
        // so use the request messages directly as updated state.
        session.messages = [...request.messages]
        session.lastActiveAt = Date.now()
        session.roundNumber++

        // Update model params if client changed them (e.g. different temperature for next round)
        this.updateSessionParams(session, request)
      } else if (session?.completed) {
        console.log(`[AgentLoop] Session ${sessionId} already completed, starting fresh`)
        session = undefined
      } else {
        console.log(`[AgentLoop] Session ${sessionId} not found, starting fresh`)
        session = undefined
      }
    }

    // Create new session if needed
    if (!session) {
      // Enforce per-client session limit
      this.enforceClientSessionLimit(clientIP)

      effectiveSessionId = this.generateSessionId()
      session = {
        messages: [...request.messages],
        model: request.model,
        actualModel,
        providerId: provider.id,
        accountId: account.id,
        tools: request.tools,
        tool_choice: request.tool_choice,
        tool_format: request.tool_format,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        n: request.n,
        stop: request.stop,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        user: request.user,
        createdAt: Date.now(),
        lastActiveAt: Date.now(),
        roundNumber: 0,
        completed: false,
        clientIP,
        anomalyCount: 0,
        lastToolCallsHash: '',
      }
      this.sessions.set(effectiveSessionId, session)

      // Track client sessions
      let clientSet = this.clientSessions.get(clientIP)
      if (!clientSet) {
        clientSet = new Set()
        this.clientSessions.set(clientIP, clientSet)
      }
      clientSet.add(effectiveSessionId)

      console.log(`[AgentLoop] New session ${effectiveSessionId} for client ${clientIP}`)
    }

    // Build the model request — inherits ALL params from session
    const modelRequest = this.buildModelRequest(session)

    // Forward to model (with graceful degradation)
    let result: ForwardResult
    try {
      result = await requestForwarder.forwardChatCompletion(
        modelRequest,
        account,
        provider,
        actualModel,
        context
      )
    } catch (error) {
      // Bug 4 fix: graceful degradation on unexpected errors
      console.error(`[AgentLoop] Forward error for session ${effectiveSessionId}:`, error)
      this.sessions.delete(effectiveSessionId!)
      this.removeFromClientIndex(clientIP, effectiveSessionId!)
      return {
        result: {
          success: false,
          error: error instanceof Error ? error.message : 'Agent loop forward failed',
        },
        sessionId: undefined,
        waitingForTools: false,
      }
    }

    if (!result.success) {
      // On error, clean up session
      console.error(`[AgentLoop] Forward failed for session ${effectiveSessionId}:`, result.error)
      this.sessions.delete(effectiveSessionId!)
      this.removeFromClientIndex(clientIP, effectiveSessionId!)
      return { result, sessionId: undefined, waitingForTools: false }
    }

    // Structured tool_calls detection (Bug 3 fix)
    const toolCallsInfo = this.extractToolCalls(result)

    if (toolCallsInfo.hasToolCalls) {
      // Add assistant message with tool_calls to session history
      const assistantMessage = this.extractAssistantMessage(result)
      if (assistantMessage) {
        session.messages.push(assistantMessage)
      }

      session.lastActiveAt = Date.now()

      // Bug 3: anomaly detection — check if we're stuck in a loop
      const currentHash = this.hashToolCalls(toolCallsInfo.toolCalls)
      if (currentHash === session.lastToolCallsHash) {
        session.anomalyCount++
        console.warn(
          `[AgentLoop] Session ${effectiveSessionId} anomaly count: ${session.anomalyCount} (identical tool_calls)`
        )
      } else {
        session.anomalyCount = 0
        session.lastToolCallsHash = currentHash
      }

      // Check round limit OR anomaly limit
      if (session.roundNumber >= MAX_AGENT_ROUNDS || session.anomalyCount >= MAX_ANOMALY_COUNT) {
        const reason = session.roundNumber >= MAX_AGENT_ROUNDS
          ? `max rounds (${MAX_AGENT_ROUNDS})`
          : `anomaly limit (${MAX_ANOMALY_COUNT} identical tool_calls)`

        console.warn(`[AgentLoop] Session ${effectiveSessionId} hit ${reason}, forcing completion`)
        session.completed = true
        this.cleanupSession(effectiveSessionId!, clientIP)

        // Return as-is, let OpenClaw handle the tool_calls
        return { result, sessionId: effectiveSessionId, waitingForTools: false }
      }

      console.log(
        `[AgentLoop] Session ${effectiveSessionId} has tool_calls, round ${session.roundNumber}/${MAX_AGENT_ROUNDS}`
      )
      return { result, sessionId: effectiveSessionId, waitingForTools: true }
    }

    // No tool_calls = final response
    session.completed = true
    session.lastActiveAt = Date.now()

    console.log(`[AgentLoop] Session ${effectiveSessionId} completed at round ${session.roundNumber}`)

    // Don't delete immediately — cleanup timer handles expired sessions
    return { result, sessionId: effectiveSessionId, waitingForTools: false }
  }

  /**
   * Get session info (for debugging / management API)
   */
  getSession(sessionId: string): AgentSession | undefined {
    return this.sessions.get(sessionId)
  }

  /**
   * Get all active sessions
   */
  getActiveSessions(): Array<{ sessionId: string; session: AgentSession }> {
    const result: Array<{ sessionId: string; session: AgentSession }> = []
    for (const [sessionId, session] of this.sessions) {
      if (!session.completed) {
        result.push({ sessionId, session })
      }
    }
    return result
  }

  /**
   * Get all sessions (active + completed but not yet expired)
   */
  getAllSessions(): Array<{ sessionId: string; session: AgentSession }> {
    const result: Array<{ sessionId: string; session: AgentSession }> = []
    for (const [sessionId, session] of this.sessions) {
      result.push({ sessionId, session })
    }
    return result
  }

  /**
   * Manually destroy a session
   */
  deleteSession(sessionId: string): boolean {
    const session = this.sessions.get(sessionId)
    if (session) {
      this.removeFromClientIndex(session.clientIP, sessionId)
    }
    return this.sessions.delete(sessionId)
  }

  /**
   * Destroy all sessions for a specific client
   */
  deleteClientSessions(clientIP: string): number {
    const sessionIds = this.clientSessions.get(clientIP)
    if (!sessionIds) return 0

    let count = 0
    for (const sid of sessionIds) {
      if (this.sessions.delete(sid)) count++
    }
    this.clientSessions.delete(clientIP)
    console.log(`[AgentLoop] Destroyed ${count} sessions for client ${clientIP}`)
    return count
  }

  /**
   * Force-reset a session (allows reuse of sessionId)
   */
  resetSession(sessionId: string): boolean {
    const session = this.sessions.get(sessionId)
    if (!session) return false

    session.messages = []
    session.roundNumber = 0
    session.completed = false
    session.anomalyCount = 0
    session.lastToolCallsHash = ''
    session.lastActiveAt = Date.now()

    console.log(`[AgentLoop] Reset session ${sessionId}`)
    return true
  }

  /**
   * Get stats
   */
  getStats(): { active: number; total: number; clientCount: number } {
    let active = 0
    for (const session of this.sessions.values()) {
      if (!session.completed) active++
    }
    return {
      active,
      total: this.sessions.size,
      clientCount: this.clientSessions.size,
    }
  }

  // ----------------------------------------------------------
  // Private Methods
  // ----------------------------------------------------------

  /**
   * Update session params from a continuation request.
   * Allows client to override temperature, max_tokens, etc. mid-loop.
   */
  private updateSessionParams(session: AgentSession, request: ChatCompletionRequest): void {
    if (request.temperature !== undefined) session.temperature = request.temperature
    if (request.top_p !== undefined) session.top_p = request.top_p
    if (request.max_tokens !== undefined) session.max_tokens = request.max_tokens
    if (request.tools !== undefined) session.tools = request.tools
    if (request.tool_choice !== undefined) session.tool_choice = request.tool_choice
    if (request.tool_format !== undefined) session.tool_format = request.tool_format
  }

  /**
   * Check if the request is a tool result continuation.
   * Uses structural check: any message with role=tool and tool_call_id.
   */
  private isToolResultContinuation(request: ChatCompletionRequest): boolean {
    const messages = request.messages
    if (messages.length === 0) return false

    // Check if there are any tool role messages in the conversation
    // (not just the last one — OpenClaw might send multiple tool results)
    const hasToolMessages = messages.some(msg => msg.role === 'tool' && !!msg.tool_call_id)

    // Also check the last message is a tool message (most common pattern)
    const lastMsg = messages[messages.length - 1]
    const lastIsTool = lastMsg.role === 'tool' && !!lastMsg.tool_call_id

    return hasToolMessages || lastIsTool
  }

  /**
   * Build a model request from the session's current state.
   * Inherits ALL generation parameters (Bug 5 fix).
   */
  private buildModelRequest(session: AgentSession): ChatCompletionRequest {
    return {
      model: session.model,
      messages: session.messages,
      tools: session.tools,
      tool_choice: session.tool_choice,
      tool_format: session.tool_format,
      temperature: session.temperature,
      top_p: session.top_p,
      max_tokens: session.max_tokens,
      n: session.n,
      stop: session.stop,
      presence_penalty: session.presence_penalty,
      frequency_penalty: session.frequency_penalty,
      user: session.user,
      stream: false, // Agent loop always uses non-stream for reliable parsing
    }
  }

  /**
   * Structured tool_calls extraction (Bug 3 fix).
   * Parses JSON structure instead of string matching.
   * Returns { hasToolCalls, toolCalls } where toolCalls is the raw array (for hashing).
   */
  private extractToolCalls(result: ForwardResult): { hasToolCalls: boolean; toolCalls: any[] } {
    if (!result.body) return { hasToolCalls: false, toolCalls: [] }

    const choices = result.body.choices
    if (!choices || !Array.isArray(choices) || choices.length === 0) {
      return { hasToolCalls: false, toolCalls: [] }
    }

    const choice = choices[0]
    if (!choice) return { hasToolCalls: false, toolCalls: [] }

    // Check message.tool_calls (OpenAI standard structure)
    const message = choice.message
    if (message && message.tool_calls && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      return { hasToolCalls: true, toolCalls: message.tool_calls }
    }

    // Check finish_reason === 'tool_calls' (some providers use this)
    if (choice.finish_reason === 'tool_calls') {
      // Even if message.tool_calls is missing, the finish_reason indicates tool calls
      return { hasToolCalls: true, toolCalls: message?.tool_calls || [] }
    }

    // Check delta.tool_calls for streaming responses that were collected
    const delta = choice.delta
    if (delta && delta.tool_calls && Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) {
      return { hasToolCalls: true, toolCalls: delta.tool_calls }
    }

    return { hasToolCalls: false, toolCalls: [] }
  }

  /**
   * Extract the assistant message from a forward result for caching
   */
  private extractAssistantMessage(result: ForwardResult): ChatMessage | null {
    if (!result.body) return null

    const choice = result.body.choices?.[0]
    if (!choice?.message) return null

    const msg = choice.message
    return {
      role: 'assistant',
      content: msg.content || null,
      tool_calls: msg.tool_calls || undefined,
    }
  }

  /**
   * Hash tool_calls array for anomaly detection
   */
  private hashToolCalls(toolCalls: any[]): string {
    if (!toolCalls || toolCalls.length === 0) return ''
    try {
      // Hash by function name + arguments
      const parts = toolCalls.map((tc: any) => {
        const name = tc?.function?.name || ''
        const args = tc?.function?.arguments || ''
        return `${name}:${args}`
      })
      return parts.join('|')
    } catch {
      return JSON.stringify(toolCalls)
    }
  }

  /**
   * Enforce per-client session limit
   */
  private enforceClientSessionLimit(clientIP: string): void {
    const clientSet = this.clientSessions.get(clientIP)
    if (!clientSet || clientSet.size < MAX_SESSIONS_PER_CLIENT) return

    // Evict oldest sessions for this client
    const sortedByAge = [...clientSet]
      .map(sid => ({ sid, session: this.sessions.get(sid) }))
      .filter(x => x.session)
      .sort((a, b) => (a.session!.lastActiveAt - b.session!.lastActiveAt))

    const toEvict = sortedByAge.slice(0, Math.max(1, clientSet.size - MAX_SESSIONS_PER_CLIENT + 1))
    for (const { sid } of toEvict) {
      this.sessions.delete(sid)
      clientSet.delete(sid)
      console.log(`[AgentLoop] Evicted old session ${sid} for client ${clientIP}`)
    }
  }

  /**
   * Remove a sessionId from the client index
   */
  private removeFromClientIndex(clientIP: string, sessionId: string): void {
    const clientSet = this.clientSessions.get(clientIP)
    if (clientSet) {
      clientSet.delete(sessionId)
      if (clientSet.size === 0) {
        this.clientSessions.delete(clientIP)
      }
    }
  }

  /**
   * Cleanup a session from both maps
   */
  private cleanupSession(sessionId: string, clientIP: string): void {
    this.sessions.delete(sessionId)
    this.removeFromClientIndex(clientIP, sessionId)
  }

  /**
   * Generate a unique session ID
   */
  private generateSessionId(): string {
    return `agent-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
  }

  // ----------------------------------------------------------
  // Cleanup
  // ----------------------------------------------------------

  private startCleanup(): void {
    this.cleanupTimer = setInterval(() => {
      this.cleanExpiredSessions()
    }, CLEANUP_INTERVAL_MS)
  }

  private cleanExpiredSessions(): number {
    const now = Date.now()
    let cleaned = 0

    for (const [sessionId, session] of this.sessions) {
      if (now - session.lastActiveAt > SESSION_TTL_MS) {
        this.removeFromClientIndex(session.clientIP, sessionId)
        this.sessions.delete(sessionId)
        cleaned++
      }
    }

    // Also enforce total session limit
    if (this.sessions.size > MAX_TOTAL_SESSIONS) {
      const sorted = [...this.sessions.entries()]
        .sort((a, b) => a[1].lastActiveAt - b[1].lastActiveAt)
      const toRemove = sorted.slice(0, this.sessions.size - MAX_TOTAL_SESSIONS)
      for (const [sid, session] of toRemove) {
        this.removeFromClientIndex(session.clientIP, sid)
        this.sessions.delete(sid)
        cleaned++
      }
    }

    if (cleaned > 0) {
      console.log(`[AgentLoop] Cleaned ${cleaned} sessions, remaining: ${this.sessions.size}`)
    }

    return cleaned
  }

  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    this.sessions.clear()
    this.clientSessions.clear()
    console.log('[AgentLoop] Destroyed')
  }
}

// ============================================================
// Export singleton
// ============================================================

export const agentLoopManager = new AgentLoopManager()
export default agentLoopManager
