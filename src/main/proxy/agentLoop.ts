/**
 * Agent Loop Manager
 * 
 * Agent 多轮工具调用循环引擎
 * 
 * 解决 Chat2API 无状态单次请求转发的原生缺陷：
 * - 当模型返回 tool_calls 时，缓存会话上下文
 * - 接收前端回传的工具执行结果，自动追加到会话历史
 * - 后端内部自动发起下一轮请求，无需用户手动发消息
 * - 递归循环直到模型不再输出工具调用、只输出普通文本
 * - 内置安全限制：最大自动轮次（默认 50 轮），防止死循环
 * 
 * 使用方式：
 * Agent 循环模式默认自动启用（只要请求体包含 tools 定义即可）
 * 如需手动关闭，可在请求头中添加 X-Agent-Mode: false
 * 首次请求返回的响应头中会包含 X-Session-Id
 * 后续工具结果回传时需携带相同的 X-Session-Id
 */

import type {
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatMessage,
  ChatCompletionTool,
  ChatCompletionToolChoice,
  ToolCall,
} from './types'

// ============================================================
// 类型定义
// ============================================================

export interface AgentSession {
  id: string
  messages: ChatMessage[]
  tools?: ChatCompletionTool[]
  toolChoice?: ChatCompletionToolChoice
  model: string
  roundCount: number
  maxRounds: number
  createdAt: number
  lastActiveAt: number
  pendingToolCalls: ToolCall[]
  status: 'active' | 'waiting_tool_result' | 'completed' | 'expired' | 'max_rounds_exceeded'
}

// ============================================================
// 常量配置
// ============================================================

const DEFAULT_MAX_ROUNDS = 50
const SESSION_TIMEOUT = 30 * 60 * 1000
const CLEANUP_INTERVAL = 60 * 1000

// ============================================================
// AgentLoopManager 核心类
// ============================================================

class AgentLoopManager {
  private sessions: Map<string, AgentSession> = new Map()
  private cleanupTimer: NodeJS.Timeout | null = null

  constructor() {
    this.startCleanup()
    console.log('[AgentLoop] Agent Loop Manager initialized')
    console.log(`[AgentLoop] Max rounds: ${DEFAULT_MAX_ROUNDS}, Session timeout: ${SESSION_TIMEOUT / 1000}s`)
  }

  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    this.sessions.clear()
    console.log('[AgentLoop] Agent Loop Manager destroyed')
  }

  private startCleanup(): void {
    this.cleanupTimer = setInterval(() => {
      const now = Date.now()
      let cleaned = 0
      for (const [id, session] of this.sessions) {
        if (now - session.lastActiveAt > SESSION_TIMEOUT) {
          session.status = 'expired'
          this.sessions.delete(id)
          cleaned++
        }
      }
      if (cleaned > 0) {
        console.log(`[AgentLoop] Cleaned up ${cleaned} expired sessions, ${this.sessions.size} active`)
      }
    }, CLEANUP_INTERVAL)
  }

  /**
   * 检查是否应该启用 Agent 循环模式
   * 
   * 默认自动启用逻辑（无需前端手动添加请求头）：
   * 1. 请求体中定义了 tools（工具列表不为空）→ 自动启用
   * 2. 除非请求头显式设置 X-Agent-Mode: false → 手动关闭
   * 3. 如果请求消息中包含 role='tool' 的消息（工具结果回传）→ 自动启用
   */
  shouldEnable(request: ChatCompletionRequest, headers: Record<string, any>): boolean {
    // 检查是否被手动关闭
    const agentModeHeader = headers['x-agent-mode']
    if (agentModeHeader === 'false') {
      console.log('[AgentLoop] Agent mode explicitly disabled via X-Agent-Mode: false')
      return false
    }

    // 自动启用条件：请求包含 tools 定义，或消息中包含 tool 角色的消息
    const hasTools = !!(request.tools && request.tools.length > 0)
    const hasToolMessages = request.messages.some(m => m.role === 'tool')
    const enabled = hasTools || hasToolMessages

    if (enabled) {
      console.log('[AgentLoop] Agent mode auto-enabled for request (hasTools=%s, hasToolMessages=%s)', hasTools, hasToolMessages)
    }
    return enabled
  }

  /**
   * 检查请求是否是工具结果回传（续接上一轮）
   * 
   * 判断条件（满足任一即可）：
   * 1. 请求头包含 X-Session-Id 且消息中有 role='tool' 的消息
   * 2. 消息中有 role='tool' 的消息，且能通过 tool_call_id 匹配到活跃会话
   */
  isContinuation(request: ChatCompletionRequest, headers: Record<string, any>): boolean {
    const sessionId = headers['x-session-id'] as string | undefined
    const hasToolMessages = request.messages.some(m => m.role === 'tool')

    if (!hasToolMessages) return false

    // 条件 1：有 X-Session-Id 头
    if (sessionId) return true

    // 条件 2：通过 tool_call_id 自动匹配活跃会话
    const toolCallId = request.messages.find(m => m.role === 'tool')?.tool_call_id
    if (toolCallId) {
      const matchedSession = this.findSessionByToolCallId(toolCallId)
      if (matchedSession) {
        console.log(`[AgentLoop] Auto-matched session ${matchedSession.id} via tool_call_id ${toolCallId}`)
        return true
      }
    }

    return false
  }

  /**
   * 解析续接请求对应的会话 ID
   * 优先使用 X-Session-Id 头，其次通过 tool_call_id 自动匹配
   */
  resolveSessionId(request: ChatCompletionRequest, headers: Record<string, any>): string | undefined {
    const sessionId = headers['x-session-id'] as string | undefined
    if (sessionId) return sessionId

    // 通过 tool_call_id 自动匹配
    const toolCallId = request.messages.find(m => m.role === 'tool')?.tool_call_id
    if (toolCallId) {
      const matchedSession = this.findSessionByToolCallId(toolCallId)
      if (matchedSession) return matchedSession.id
    }

    return undefined
  }

  generateSessionId(): string {
    return `agent-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }

  createSession(sessionId: string, request: ChatCompletionRequest): AgentSession {
    const session: AgentSession = {
      id: sessionId,
      messages: request.messages.map(m => ({ ...m })),
      tools: request.tools ? request.tools.map(t => ({ ...t })) : undefined,
      toolChoice: request.tool_choice,
      model: request.model,
      roundCount: 0,
      maxRounds: DEFAULT_MAX_ROUNDS,
      createdAt: Date.now(),
      lastActiveAt: Date.now(),
      pendingToolCalls: [],
      status: 'active',
    }
    this.sessions.set(sessionId, session)
    console.log(`[AgentLoop] Session created: ${sessionId}`)
    return session
  }

  getSession(sessionId: string): AgentSession | undefined {
    const session = this.sessions.get(sessionId)
    if (session) {
      session.lastActiveAt = Date.now()
    }
    return session
  }

  findSessionByToolCallId(toolCallId: string): AgentSession | undefined {
    for (const session of this.sessions.values()) {
      if (session.pendingToolCalls.some(tc => tc.id === toolCallId)) {
        return session
      }
    }
    return undefined
  }

  deleteSession(sessionId: string): boolean {
    const deleted = this.sessions.delete(sessionId)
    if (deleted) {
      console.log(`[AgentLoop] Session deleted: ${sessionId}`)
    }
    return deleted
  }

  appendToolResults(sessionId: string, request: ChatCompletionRequest): AgentSession | null {
    const session = this.sessions.get(sessionId)
    if (!session) {
      console.warn(`[AgentLoop] Session not found for tool results: ${sessionId}`)
      const toolCallId = request.messages.find(m => m.role === 'tool')?.tool_call_id
      if (toolCallId) {
        const foundSession = this.findSessionByToolCallId(toolCallId)
        if (foundSession) {
          console.log(`[AgentLoop] Found session ${foundSession.id} via tool_call_id ${toolCallId}`)
          return this.appendToolResults(foundSession.id, request)
        }
      }
      return null
    }

    let appendedCount = 0
    for (const msg of request.messages) {
      if (msg.role === 'tool') {
        session.messages.push({ ...msg })
        appendedCount++
      }
    }

    session.pendingToolCalls = []
    session.status = 'active'
    session.lastActiveAt = Date.now()

    console.log(`[AgentLoop] Appended ${appendedCount} tool results to session ${sessionId}`)
    return session
  }

  processModelResponse(sessionId: string, response: any): boolean {
    const session = this.sessions.get(sessionId)
    if (!session) {
      console.warn(`[AgentLoop] Session not found when processing response: ${sessionId}`)
      return false
    }

    session.roundCount++
    session.lastActiveAt = Date.now()

    const message = response?.choices?.[0]?.message
    const toolCalls = message?.tool_calls

    if (toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0) {
      session.messages.push({
        role: 'assistant',
        content: message?.content || null,
        tool_calls: toolCalls.map(tc => ({ ...tc })),
      } as ChatMessage)

      session.pendingToolCalls = toolCalls.map(tc => ({ ...tc }))
      session.status = 'waiting_tool_result'

      console.log(
        `[AgentLoop] Round ${session.roundCount}/${session.maxRounds}: ` +
        `Model returned ${toolCalls.length} tool call(s): ` +
        toolCalls.map(tc => tc.function?.name).join(', ')
      )
      return true
    }

    session.messages.push({
      role: 'assistant',
      content: message?.content || '',
    })

    session.status = 'completed'
    console.log(
      `[AgentLoop] Round ${session.roundCount}/${session.maxRounds}: ` +
      `Final text response received, loop completed`
    )

    this.sessions.delete(sessionId)
    return false
  }

  buildNextRequest(session: AgentSession, originalRequest: ChatCompletionRequest): ChatCompletionRequest {
    return {
      ...originalRequest,
      messages: session.messages.map(m => ({ ...m })),
      tools: session.tools,
      tool_choice: session.toolChoice,
      stream: false,
    }
  }

  isMaxRoundsExceeded(sessionId: string): boolean {
    const session = this.sessions.get(sessionId)
    if (!session) return false
    return session.roundCount >= session.maxRounds
  }

  getRoundCount(sessionId: string): number {
    const session = this.sessions.get(sessionId)
    return session?.roundCount || 0
  }

  getMaxRounds(sessionId: string): number {
    const session = this.sessions.get(sessionId)
    return session?.maxRounds || DEFAULT_MAX_ROUNDS
  }

  getActiveSessionCount(): number {
    return this.sessions.size
  }

  static hasToolCalls(response: any): boolean {
    const toolCalls = response?.choices?.[0]?.message?.tool_calls
    return !!(toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0)
  }

  static extractToolCalls(response: any): ToolCall[] {
    const toolCalls = response?.choices?.[0]?.message?.tool_calls
    return (toolCalls && Array.isArray(toolCalls)) ? toolCalls : []
  }
}

export const agentLoopManager = new AgentLoopManager()
export default agentLoopManager
