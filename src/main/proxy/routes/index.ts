/**
 * Proxy Service Module - Route Index
 * Export all routes
 */

import chatRouter from './chat'
import modelsRouter from './models'
import completionsRouter from './completions'
import claudeRouter from '../claude/route'

export {
  chatRouter,
  modelsRouter,
  completionsRouter,
  claudeRouter,
}

export default [
  chatRouter,
  modelsRouter,
  completionsRouter,
  claudeRouter,
]
