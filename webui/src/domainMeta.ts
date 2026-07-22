/** 域中文标注与回测分轨分组 */

export interface DomainMeta {
  key: string
  label: string
  desc: string
  group: string
}

export const DOMAIN_GROUPS = [
  '因子侧回测',
  '策略侧回测',
  '因子挖掘',
  '风险模型',
] as const

export type DomainGroup = (typeof DOMAIN_GROUPS)[number] | '其他'

export const DOMAIN_META: Record<string, DomainMeta> = {
  factor_evaluations: {
    key: 'factor_evaluations',
    label: '因子评估',
    desc: '信号轨 IC/分层评估与单因子交易轨回测产物',
    group: '因子侧回测',
  },
  combinations: {
    key: 'combinations',
    label: '组合分数面板',
    desc: '多因子合成的组合分数,供组合回测消费',
    group: '因子侧回测',
  },
  combine_backtests: {
    key: 'combine_backtests',
    label: '组合回测',
    desc: '多因子组合分数的交易轨净值回测',
    group: '因子侧回测',
  },
  strategies: {
    key: 'strategies',
    label: '策略回测',
    desc: '规则型策略(择时/轮动/分层建仓)的预置权重回测',
    group: '策略侧回测',
  },
  sim: {
    key: 'sim',
    label: '模拟交易',
    desc: '组合优化器目标权重的模拟撮合回测',
    group: '策略侧回测',
  },
  execution: {
    key: 'execution',
    label: '向前执行',
    desc: '逐日向前纸面撮合与持仓状态',
    group: '策略侧回测',
  },
  portfolios: {
    key: 'portfolios',
    label: '组合优化',
    desc: '组合优化器落盘的目标权重产物',
    group: '策略侧回测',
  },
  mining_sessions: {
    key: 'mining_sessions',
    label: '挖掘会话',
    desc: '单 Agent 因子挖掘会话记录',
    group: '因子挖掘',
  },
  mine_team: {
    key: 'mine_team',
    label: '团队挖掘',
    desc: '4 角色 LLM 团队挖掘会话',
    group: '因子挖掘',
  },
  mine_agent: {
    key: 'mine_agent',
    label: 'Agent 挖掘',
    desc: 'LLM 单 Agent 挖掘产物',
    group: '因子挖掘',
  },
  risk_models: {
    key: 'risk_models',
    label: '风险模型',
    desc: 'Barra 风格/行业暴露与协方差产物',
    group: '风险模型',
  },
}

/** 取域中文名；未知域回退原 key */
export function domainLabel(key: string): string {
  return DOMAIN_META[key]?.label ?? key
}

/** 取域说明；未知域空串 */
export function domainDesc(key: string): string {
  return DOMAIN_META[key]?.desc ?? ''
}

/** 取域分组；未知域 → 其他 */
export function domainGroup(key: string): DomainGroup {
  const g = DOMAIN_META[key]?.group
  if (g && (DOMAIN_GROUPS as readonly string[]).includes(g)) {
    return g as DomainGroup
  }
  return '其他'
}

/**
 * 将 domain 列表按 DOMAIN_GROUPS 分组，未知域归入「其他」。
 * 组内顺序跟随 domains 原序。
 */
export function groupDomains(domains: string[]): {
  group: DomainGroup
  keys: string[]
}[] {
  const buckets = new Map<DomainGroup, string[]>()
  for (const g of DOMAIN_GROUPS) {
    buckets.set(g, [])
  }
  buckets.set('其他', [])

  for (const key of domains) {
    const g = domainGroup(key)
    buckets.get(g)!.push(key)
  }

  const result: { group: DomainGroup; keys: string[] }[] = []
  for (const g of DOMAIN_GROUPS) {
    const keys = buckets.get(g)!
    if (keys.length > 0) result.push({ group: g, keys })
  }
  const other = buckets.get('其他')!
  if (other.length > 0) result.push({ group: '其他', keys: other })
  return result
}
