import Router from '@koa/router'
import type { Context } from 'koa'
import ConfigManager from '../../../store/config'
import { managementAuthMiddleware } from '../../middleware/managementAuth'
import {
  buildSmokeFixture,
  getLatestToolCallingSmokeResult,
  setLatestToolCallingSmokeResult,
} from '../../toolCalling/diagnostics'
import { agentLoopManager } from '../../agentLoop'
import { loadBalancer } from '../../loadbalancer'
import { modelMapper } from '../../modelMapper'
import { storeManager } from '../../../store/store'
import type { ManagementApiResponse } from '../../../../shared/types'
import type { ToolClientAdapterId } from '../../../../shared/toolCalling'

const router = new Router({ prefix: '/v0/management/tool-calling' })

router.use(managementAuthMiddleware)

router.get('/status', async (ctx: Context) => {
  const config = ConfigManager.get()

  ctx.body = {
    success: true,
    data: {
      config: config.toolCallingConfig,
      latestSmokeResult: getLatestToolCallingSmokeResult(),
    },
  } as ManagementApiResponse
})

router.post('/smoke', async (ctx: Context) => {
  const body = ctx.request.body as { clientAdapterId?: ToolClientAdapterId; model?: string }
  const config = ConfigManager.get()
  const clientAdapterId = body.clientAdapterId ?? config.toolCallingConfig.clientAdapterId
  const fixture = buildSmokeFixture(clientAdapterId)

  // Try to run a live smoke test through the agent loop if a model is available
  const testModel = body.model
  if (testModel) {
    try {
      const preferredProviderId = modelMapper.getPreferredProvider(testModel)
      const preferredAccountId = modelMapper.getPreferredAccount(testModel)
      const selection = loadBalancer.selectAccount(
        testModel,
        config.loadBalanceStrategy,
        preferredProviderId,
        preferredAccountId
      )

      if (selection) {
        const { account, provider, actualModel } = selection
        const smokeRequest = {
          ...fixture,
          model: testModel,
          stream: false,
        }

        const result = await agentLoopManager.handleRequest(
          smokeRequest as any,
          account,
          provider,
          actualModel,
          {
            requestId: `smoke-${Date.now()}`,
            providerId: provider.id,
            accountId: account.id,
            model: testModel,
            actualModel,
            startTime: Date.now(),
            isStream: false,
            clientIP: 'smoke-test',
          },
          undefined,
          'smoke-test'
        )

        if (result.result.success) {
          const choice = result.result.body?.choices?.[0]
          const hasToolCalls = !!(choice?.message?.tool_calls?.length)
          const smokeResult = setLatestToolCallingSmokeResult({
            success: true,
            category: hasToolCalls ? 'pass' : 'model_did_not_call_tool',
            message: hasToolCalls
              ? `Live smoke passed: model returned ${choice.message.tool_calls.length} tool call(s).`
              : 'Model responded but did not invoke tools. Tool calling may not be supported by this model.',
            clientAdapterId,
            providerId: provider.id,
            requestId: result.result.body?.id,
            timestamp: Date.now(),
          })

          ctx.body = {
            success: true,
            data: {
              result: smokeResult,
              fixture,
              liveTest: true,
              sessionId: result.sessionId,
              waitingForTools: result.waitingForTools,
            },
          } as ManagementApiResponse
          return
        } else {
          const smokeResult = setLatestToolCallingSmokeResult({
            success: false,
            category: 'provider_or_account_error',
            message: `Live smoke failed: ${result.result.error}`,
            clientAdapterId,
            providerId: provider.id,
            timestamp: Date.now(),
          })
          ctx.body = {
            success: false,
            data: {
              result: smokeResult,
              fixture,
              liveTest: true,
            },
          } as ManagementApiResponse
          return
        }
      }
    } catch (error) {
      const smokeResult = setLatestToolCallingSmokeResult({
        success: false,
        category: 'provider_or_account_error',
        message: `Live smoke exception: ${error instanceof Error ? error.message : 'Unknown error'}`,
        clientAdapterId,
        timestamp: Date.now(),
      })
      ctx.body = {
        success: false,
        data: {
          result: smokeResult,
          fixture,
          liveTest: true,
        },
      } as ManagementApiResponse
      return
    }
  }

  // No model specified — return fixture only (offline smoke)
  const result = setLatestToolCallingSmokeResult({
    success: true,
    category: 'pass',
    message: 'Smoke fixture generated. Pass a "model" field to run a live provider smoke test.',
    clientAdapterId,
    timestamp: Date.now(),
  })

  ctx.body = {
    success: true,
    data: {
      result,
      fixture,
      liveTest: false,
    },
  } as ManagementApiResponse
})

export default router
