/** API 响应类型 */

export interface HealthResponse {
  status: string
  domains: string[]
}

export interface LatestRun {
  run_id: string
  status: string | null
  git_sha: string | null
}

export interface DomainOverview {
  domain: string
  count: number
  latest: LatestRun | null
}

export interface OverviewResponse {
  domains: DomainOverview[]
}

export interface RunSummary {
  run_id: string
  domain: string
  git_sha: string | null
  status: string | null
  manifest: Record<string, unknown>
}

export interface RunsResponse {
  domain: string
  runs: RunSummary[]
}

export interface RunDetailResponse {
  run_id: string
  domain: string
  manifest: Record<string, unknown>
  metrics?: Record<string, unknown>
}

export interface NavResponse {
  domain: string
  run_id: string
  nav: [string, number][]
}
