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
 * - 内置安全限制：最大自动轮次（默认 15 轮），防止死循环
 * 
 * 使用方式：
 * 前端在请求头中添加 X-Agent-Mode: true 即可激活 Agent 循环模式
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

/**
 * Agent 会话 - 存储一次多轮工具调用循环的完整上下文
 */
export interface AgentSession {
  /** 会话唯一标识 */
  id: string
  /** 完整对话消息历史（包含用户消息、助手回复、工具调用、工具结果） */
  messages: ChatMessage[]
  /** 工具定义列表 */
  tools?: ChatCompletionTool[]
  /** 工具选择策略 */
  toolChoice?: ChatCompletionToolChoice
  /** 请求的模型名称 */
  model: string
  /** 当前已执行的轮次数 */
  roundCount: number
  /** 最大允许轮次数 */
  maxRounds: number
  /** 会话创建时间 */
  createdAt: number
  /** 最后活跃时间（用于过期清理） */
  lastActiveAt: number
  /** 当前待处理的工具调用（等待前端回传执行结果） */
  pendingToolCalls: ToolCall[]
  /** 会话状态 */
  status: 'active' | 'waiting_tool_result' | 'completed' | 'expired' | 'max_rounds_exceeded'
}

/**
 * Agent 循环处理结果
 */
export interface AgentLoopResult {
  /** 是否触发了 Agent 循环 */
  agentLoopActive: boolean
  /** 会话 ID */
  sessionId: string
  /** 是否包含待处理的工具调用（需要前端执行并回传结果） */
  hasPendingToolCalls: boolean
  /** 当前轮次 */
  roundCount: number
  /** 是否已达到最大轮次 */
  maxRoundsExceeded: boolean
}

// ============================================================
// 常量配置
// ============================================================

/** 默认最大自动轮次 */
const DEFAULT_MAX_ROUNDS = 15

/** 会话超时时间（30 分钟） */
const SESSION_TIMEOUT = 30 * 60 * 1000

/** 会话清理检查间隔（1 分钟） */
const CLEANUP_INTERVAL = 60 * 1000

// ============================================================
// AgentLoopManager 核心类
// ============================================================

class AgentLoopManager {
  /** 内存会话缓存：sessionId -> AgentSession */
  private sessions: Map<string, AgentSession> = new Map()

  /** 过期清理定时器 */
  private cleanupTimer: NodeJS.Timeout | null = null

  constructor() {
    this.startCleanup()
    console.log('[AgentLoop] Agent Loop Manager initialized')
    console.log(`[AgentLoop] Max rounds: ${DEFAULT_MAX_ROUNDS}, Session timeout: ${SESSION_TIMEOUT / 1000}s`)
  }

  /**
   * 销毁管理器，清理资源
   */
  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer)
      this.cleanupTimer = null
    }
    this.sessions.clear()
    console.log('[AgentLoop] Agent Loop Manager destroyed')
  }

  // ============================================================
  // 会话生命周期管理
  // ============================================================

  /**
   * 启动过期会话清理定时器
   */
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
   * 启用条件：
   * 1. 请求头包含 X-Agent-Mode: true
   * 2. 请求体中定义了 tools（工具列表不为空）
   */
  shouldEnable(request: ChatCompletionRequest, headers: Record<string, any>): boolean {
    const agentModeHeader = headers['x-agent-mode']
    const hasTools = !!(request.tools && request.tools.length > 0)
    const enabled = agentModeHeader === 'true' && hasTools
    if (enabled) {
      console.log('[AgentLoop] Agent mode enabled for request')
    }
    return enabled
  }

  /**
   * 检查请求是否是工具结果回传（续接上一轮）
   * 
   * 判断条件：
   * 1. 请求头包含 X-Session-Id
   * 2. 请求消息中包含 role='tool' 的消息
   */
  isContinuation(request: ChatCompletionRequest, headers: Record<string, any>): boolean {
    const sessionId = headers['x-session-id'] as string | undefined
    const hasToolMessages = request.messages.some(m => m.role === 'tool')
    return !!(sessionId && hasToolMessages)
  }

  /**
   * 生成新的会话 ID
   */
  generateSessionId(): string {
    return `agent-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  }

  /**
   * 创建新会话
   */
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

  /**
   * 获取会话
   */
  getSession(sessionId: string): AgentSession | undefined {
    const session = this.sessions.get(sessionId)
    if (session) {
      session.lastActiveAt = Date.now()
    }
    return session
  }

  /**
   * 通过 tool_call_id 查找会话（兼容不发送 X-Session-Id 的客户端）
   */
  findSessionByToolCallId(toolCallId: string): AgentSession | undefined {
    for (const session of this.sessions.values()) {
      if (session.pendingToolCalls.some(tc => tc.id === toolCallId)) {
        return session
      }
    }
    return undefined
  }

  /**
   * 删除会话
   */
  deleteSession(sessionId: string): boolean {
    const deleted = this.sessions.delete(sessionId)
    if (deleted) {
      console.log(`[AgentLoop] Session deleted: ${sessionId}`)
    }
    return deleted
  }

  // ============================================================
  // 消息处理
  // ============================================================

  /**
   * 将工具执行结果追加到会话历史中
   * 
   * @param sessionId 会话 ID
   * @param request 包含工具结果的请求
   * @returns 更新后的会话，如果会话不存在则返回 null
   */
  appendToolResults(sessionId: string, request: ChatCompletionRequest): AgentSession | null {
    const session = this.sessions.get(sessionId)
    if (!session) {
      console.warn(`[AgentLoop] Session not found for tool results: ${sessionId}`)
      // 尝试通过 tool_call_id 查找会话
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

    // 追加所有 tool 角色的消息到会话历史
    let appendedCount = 0
    for (const msg of request.messages) {
      if (msg.role === 'tool') {
        session.messages.push({ ...msg })
        appendedCount++
      }
    }

    // 清除待处理的工具调用
    session.pendingToolCalls = []
    session.status = 'active'
    session.lastActiveAt = Date.now()

    console.log(`[AgentLoop] Appended ${appendedCount} tool results to session ${sessionId}`)
    return session
  }

  /**
   * 处理模型响应，更新会话状态
   * 
   * @param sessionId 会话 ID
   * @param response 模型响应体
   * @returns true 表示响应包含 tool_calls（需要继续循环），false 表示最终文本回复
   */
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
      // 模型返回了工具调用 → 存储到会话，等待前端执行
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
      return true // 有 tool_calls，循环继续
    }

    // 模型返回了最终文本回复 → 循环结束
    session.messages.push({
      role: 'assistant',
      content: message?.content || '',
    })

    session.status = 'completed'
    console.log(
      `[AgentLoop] Round ${session.roundCount}/${session.maxRounds}: ` +
      `Final text response received, loop completed`
    )

    // 清理会话
    this.sessions.delete(sessionId)
    return false // 无 tool_calls，循环结束
  }

  // ============================================================
  // 请求构建
  // ============================================================

  /**
   * 基于会话状态构建下一轮请求
   * 
   * 使用会话中缓存的完整消息历史、工具定义和工具选择策略
   * 强制使用非流式模式（Agent 循环需要完整响应来判断是否有 tool_calls）
   */
  buildNextRequest(session: AgentSession, originalRequest: ChatCompletionRequest): ChatCompletionRequest {
    return {
      ...originalRequest,
      messages: session.messages.map(m => ({ ...m })),
      tools: session.tools,
      tool_choice: session.toolChoice,
      stream: false, // Agent 循环内部强制非流式，确保能完整解析 tool_calls
    }
  }

  // ============================================================
  // 安全限制
  // ============================================================

  /**
   * 检查是否已超过最大轮次
   */
  isMaxRoundsExceeded(sessionId: string): boolean {
    const session = this.sessions.get(sessionId)
    if (!session) return false
    return session.roundCount >= session.maxRounds
  }

  /**
   * 获取当前轮次
   */
  getRoundCount(sessionId: string): number {
    const session = this.sessions.get(sessionId)
    return session?.roundCount || 0
  }

  /**
   * 获取最大轮次
   */
  getMaxRounds(sessionId: string): number {
    const session = this.sessions.get(sessionId)
    return session?.maxRounds || DEFAULT_MAX_ROUNDS
  }

  // ============================================================
  // 状态查询
  // ============================================================

  /**
   * 获取当前活跃会话数量
   */
  getActiveSessionCount(): number {
    return this.sessions.size
  }

  /**
   * 获取所有活跃会话的摘要信息
   */
  getSessionSummaries(): Array<{
    id: string
    model: string
    roundCount: number
    maxRounds: number
    status: string
    pendingToolCalls: number
    createdAt: number
    lastActiveAt: number
  }> {
    return Array.from(this.sessions.values()).map(session => ({
      id: session.id,
      model: session.model,
      roundCount: session.roundCount,
      maxRounds: session.maxRounds,
      status: session.status,
      pendingToolCalls: session.pendingToolCalls.length,
      createdAt: session.createdAt,
      lastActiveAt: session.lastActiveAt,
    }))
  }

  /**
   * 检查响应是否包含 tool_calls（不依赖会话状态的静态方法）
   */
  static hasToolCalls(response: any): boolean {
    const toolCalls = response?.choices?.[0]?.message?.tool_calls
    return !!(toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0)
  }

  /**
   * 从响应中提取 tool_calls
   */
  static extractToolCalls(response: any): ToolCall[] {
    const toolCalls = response?.choices?.[0]?.message?.tool_calls
    return (toolCalls && Array.isArray(toolCalls)) ? toolCalls : []
  }
}

// ============================================================
// 导出单例
// ============================================================

export const agentLoopManager = new AgentLoopManager()
export default agentLoopManager
