/**
 * Proxy Service Module - Models Route
 * Implements /v1/models route
 */

import Router from '@koa/router'
import type { Context } from 'koa'
import { ModelsResponse, ModelInfo } from '../types'
import { loadBalancer } from '../loadbalancer'
import { storeManager } from '../../store/store'
import { modelMapper } from '../modelMapper'

const router = new Router({ prefix: '/v1' })

/**
 * Built-in Claude models for Claude Code compatibility
 * These models are always available in /v1/models so that
 * Claude Code and other Anthropic-compatible clients can discover them.
 */
const CLAUDE_BUILTIN_MODELS: ModelInfo[] = [
  { id: 'claude-opus-4-20250514', object: 'model', created: 1747267200, owned_by: 'anthropic' },
  { id: 'claude-sonnet-4-20250514', object: 'model', created: 1747267200, owned_by: 'anthropic' },
  { id: 'claude-sonnet-4-5-20250514', object: 'model', created: 1747267200, owned_by: 'anthropic' },
  { id: 'claude-haiku-4-5-20251001', object: 'model', created: 1759334400, owned_by: 'anthropic' },
  { id: 'claude-3-5-sonnet-20241022', object: 'model', created: 1729555200, owned_by: 'anthropic' },
  { id: 'claude-3-5-haiku-20241022', object: 'model', created: 1729555200, owned_by: 'anthropic' },
  { id: 'claude-3-opus-20240229', object: 'model', created: 1709164800, owned_by: 'anthropic' },
]

/**
 * Get all available models
 */
router.get('/models', async (ctx: Context) => {
  const providers = storeManager.getProviders().filter(p => p.enabled)
  const models: ModelInfo[] = []
  const addedModels = new Set<string>()

  for (const provider of providers) {
    const accounts = storeManager.getAccountsByProviderId(provider.id)
      .filter(account => account.status === 'active')

    if (accounts.length === 0) {
      continue
    }

    const effectiveModels = storeManager.getEffectiveModels(provider.id)
    for (const model of effectiveModels) {
      if (!addedModels.has(model.displayName)) {
        addedModels.add(model.displayName)
        models.push({
          id: model.displayName,
          object: 'model',
          created: Math.floor(provider.createdAt / 1000),
          owned_by: provider.name,
        })
      }
    }
  }

  const config = storeManager.getConfig()
  const mappings = config.modelMappings || {}
  for (const [requestModel, mapping] of Object.entries(mappings)) {
    if (!addedModels.has(requestModel)) {
      addedModels.add(requestModel)
      models.push({
        id: requestModel,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'model-mapping',
      })
    }
  }

  // Add built-in Claude models for Claude Code compatibility
  for (const claudeModel of CLAUDE_BUILTIN_MODELS) {
    if (!addedModels.has(claudeModel.id)) {
      addedModels.add(claudeModel.id)
      models.push(claudeModel)
    }
  }

  const response: ModelsResponse = {
    object: 'list',
    data: models,
  }

  ctx.set('Content-Type', 'application/json')
  ctx.body = response
})

/**
 * Get specified model info
 */
router.get('/models/:model', async (ctx: Context) => {
  const modelId = ctx.params.model

  // Check built-in Claude models first
  const claudeModel = CLAUDE_BUILTIN_MODELS.find(m => m.id === modelId)
  if (claudeModel) {
    ctx.set('Content-Type', 'application/json')
    ctx.body = claudeModel
    return
  }

  const config = storeManager.getConfig()
  const mappings = config.modelMappings || {}
  if (mappings[modelId]) {
    ctx.set('Content-Type', 'application/json')
    ctx.body = {
      id: modelId,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'model-mapping',
    }
    return
  }

  const providers = storeManager.getProviders().filter(p => p.enabled)

  for (const provider of providers) {
    const accounts = storeManager.getAccountsByProviderId(provider.id)
      .filter(account => account.status === 'active')

    if (accounts.length === 0) {
      continue
    }

    const effectiveModels = storeManager.getEffectiveModels(provider.id)
    const normalizedModelId = modelId.toLowerCase()
    const found = effectiveModels.some(m => {
      const normalizedSupported = m.displayName.toLowerCase()
      if (normalizedSupported.endsWith('*')) {
        return normalizedModelId.startsWith(normalizedSupported.slice(0, -1))
      }
      return normalizedSupported === normalizedModelId
    })

    if (found) {
      ctx.set('Content-Type', 'application/json')
      ctx.body = {
        id: modelId,
        object: 'model',
        created: Math.floor(provider.createdAt / 1000),
        owned_by: provider.name,
      }
      return
    }
  }

  ctx.status = 404
  ctx.body = {
    error: {
      message: `Model '${modelId}' not found`,
      type: 'invalid_request_error',
      param: 'model',
      code: 'model_not_found',
    },
  }
})

export default router
