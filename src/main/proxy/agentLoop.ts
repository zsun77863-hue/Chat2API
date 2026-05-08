/**
 * Agent Loop Module
 * 
 * Solves Chat2API's stateless architecture problem by adding session-based
 * conversation caching and automatic multi-turn tool-calling loop support.
 * 
 * Flow:
 * 1. OpenClaw sends initial request → forward to model → if tool_calls, cache session & return
 * 2. OpenClaw sends tool results (role=tool messages) → merge with cached session → forward to model
 * 3. Repeat until model returns text-only response (no tool_calls) → return final result & cleanup
 * 
 * Safety: max 15 auto-rounds to prevent infinite loops.
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
  /** Temperature and other generation params */
  temperature?: number
  top_p?: number
  max_tokens?: number
  /** Session creation time */
  createdAt: number
  /** Last activity time */
  lastActiveAt: number
  /** Current round number (0 = initial request) */
  roundNumber: number
  /** Whether this session has been completed (model returned final text) */
  completed: boolean
}

// ============================================================
// Configuration
// ============================================================

const MAX_AGENT_ROUNDS = 15
const SESSION_TTL_MS = 30 * 60 * 1000 // 30 minutes
const CLEANUP_INTERVAL_MS = 60 * 1000 // 1 minute

// ============================================================
// Agent Loop Manager (Singleton)
// ============================================================

class AgentLoopManager {
  /** Map of sessionId → AgentSession */
  private sessions: Map<string, AgentSession> = new Map()
  private cleanupTimer: NodeJS.Timeout | null = null

  constructor() {
    this.startCleanup()
    console.log('[AgentLoop] Initialized, max rounds:', MAX_AGENT_ROUNDS)
  }

  // ----------------------------------------------------------
  // Public API
  // ----------------------------------------------------------

  /**
   * Main entry point: handle a chat completion request with agent loop support.
   * 
   * Returns a sessionId in the response headers if a session was created/cached,
   * so the caller can track it.
   */
  async handleRequest(
    request: ChatCompletionRequest,
    account: Account,
    provider: Provider,
    actualModel: string,
    context: ProxyContext,
    sessionId?: string
  ): Promise<{ result: ForwardResult; sessionId?: string }> {
    const roundNumber = this.detectRoundNumber(request)

    // Check if this is a continuation (has tool result messages)
    const isToolResultContinuation = this.isToolResultContinuation(request)

    let session: AgentSession | undefined
    let effectiveSessionId: string | undefined

    if (isToolResultContinuation && sessionId) {
      // Try to find existing session
      session = this.sessions.get(sessionId)
      if (session && !session.completed) {
        effectiveSessionId = sessionId
        console.log(`[AgentLoop] Continuing session ${sessionId}, round ${session.roundNumber + 1}`)

        // OpenClaw sends the FULL conversation history in each request,
        // so we just use the new request's messages directly as the updated state.
        // No merge needed — the request already contains:
        //   system + original user msg + assistant(tool_calls) + tool results
        session.messages = [...request.messages]
        session.lastActiveAt = Date.now()
        session.roundNumber++
      } else if (session?.completed) {
        console.log(`[AgentLoop] Session ${sessionId} already completed, starting fresh`)
        session = undefined
      }
    }

    // Create new session if needed
    if (!session) {
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
        createdAt: Date.now(),
        lastActiveAt: Date.now(),
        roundNumber: 0,
        completed: false,
      }
      this.sessions.set(effectiveSessionId!, session)
      console.log(`[AgentLoop] New session ${effectiveSessionId}`)
    }

    // Build the request to send to the model
    const modelRequest = this.buildModelRequest(session)

    // Forward to model
    const result = await requestForwarder.forwardChatCompletion(
      modelRequest,
      account,
      provider,
      actualModel,
      context
    )

    if (!result.success) {
      // On error, clean up session
      this.sessions.delete(effectiveSessionId!)
      return { result, sessionId: undefined }
    }

    // Check if response contains tool_calls
    const hasToolCalls = this.responseHasToolCalls(result)

    if (hasToolCalls) {
      // Add assistant message with tool_calls to session history
      const assistantMessage = this.extractAssistantMessage(result)
      if (assistantMessage) {
        session.messages.push(assistantMessage)
      }

      session.lastActiveAt = Date.now()

      // Check round limit
      if (session.roundNumber >= MAX_AGENT_ROUNDS) {
        console.warn(`[AgentLoop] Session ${effectiveSessionId} hit max rounds (${MAX_AGENT_ROUNDS}), forcing completion`)
        session.completed = true
        this.sessions.delete(effectiveSessionId!)

        // Return as-is, let OpenClaw handle the tool_calls
        return { result, sessionId: effectiveSessionId }
      }

      console.log(`[AgentLoop] Session ${effectiveSessionId} has tool_calls, round ${session.roundNumber}/${MAX_AGENT_ROUNDS}`)
      return { result, sessionId: effectiveSessionId }
    }

    // No tool_calls = final response
    session.completed = true
    session.lastActiveAt = Date.now()

    console.log(`[AgentLoop] Session ${effectiveSessionId} completed at round ${session.roundNumber}`)

    // Don't delete immediately - OpenClaw might need the sessionId for reference
    // The cleanup timer will handle expired sessions

    return { result, sessionId: effectiveSessionId }
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
   * Manually clean up a session
   */
  deleteSession(sessionId: string): boolean {
    return this.sessions.delete(sessionId)
  }

  /**
   * Get stats
   */
  getStats(): { active: number; total: number } {
    let active = 0
    for (const session of this.sessions.values()) {
      if (!session.completed) active++
    }
    return { active, total: this.sessions.size }
  }

  // ----------------------------------------------------------
  // Private Methods
  // ----------------------------------------------------------

  /**
   * Detect which round number this request represents
   * by counting assistant+tool message pairs in the request
   */
  private detectRoundNumber(request: ChatCompletionRequest): number {
    let round = 0
    for (const msg of request.messages) {
      if (msg.role === 'assistant' && msg.tool_calls) {
        round++
      }
    }
    return round
  }

  /**
   * Check if the request is a tool result continuation
   * (last message has role=tool)
   */
  private isToolResultContinuation(request: ChatCompletionRequest): boolean {
    const messages = request.messages
    if (messages.length === 0) return false
    const lastMsg = messages[messages.length - 1]
    return lastMsg.role === 'tool' && !!lastMsg.tool_call_id
  }

  /**
   * Build a model request from the session's current state
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
      stream: false, // Agent loop always uses non-stream for easier handling
    }
  }

  /**
   * Check if a forward result contains tool_calls
   */
  private responseHasToolCalls(result: ForwardResult): boolean {
    if (!result.body) return false

    // Check non-stream response
    const choices = result.body.choices
    if (!choices || choices.length === 0) return false

    const choice = choices[0]
    const message = choice.message
    if (!message) return false

    // Check for tool_calls in the message
    if (message.tool_calls && message.tool_calls.length > 0) {
      return true
    }

    // Check finish_reason
    if (choice.finish_reason === 'tool_calls') {
      return true
    }

    return false
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
        this.sessions.delete(sessionId)
        cleaned++
      }
    }

    if (cleaned > 0) {
      console.log(`[AgentLoop] Cleaned ${cleaned} expired sessions`)
    }

    return cleaned
  }

  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    this.sessions.clear()
    console.log('[AgentLoop] Destroyed')
  }
}

// ============================================================
// Export singleton
// ============================================================

export const agentLoopManager = new AgentLoopManager()
export default agentLoopManager
