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

// ---- 因子库 ----

export type Market = 'ashare' | 'crypto' | 'us' | 'futures'

export interface FactorRecord {
  expression?: string
  market?: string
  status?: string | null
  ic_train?: number | null
  holdout_ic?: number | null
  dsr?: number | null
  turnover?: number | null
  admission_track?: string | null
  [key: string]: unknown
}

export interface LibraryResponse {
  market: string
  count: number
  by_status: Record<string, number>
  factors: FactorRecord[]
}

export interface TrackPoint {
  date: string | null
  ic: number | null
  n_stocks: number | null
}

export interface TrackResponse {
  expression: string
  points: TrackPoint[]
}

// ---- 因子资产 ----

export interface StoreEntry {
  name: string
  kind?: string
  expression?: string
  frequency?: string
  description?: string
  source_run_id?: string
  created_at?: string
  ledger_snapshot?: {
    status?: string | null
    ic_train?: number | null
    holdout_ic?: number | null
    [key: string]: unknown
  } | null
  [key: string]: unknown
}

export interface StoreListResponse {
  market: string
  entries: StoreEntry[]
}

export interface StoreDetailResponse {
  market: string
  name: string
  meta: StoreEntry
  source: string | null
}

// ---- 运营 ----

export interface CampaignSummary {
  name: string
  done: boolean
  exitcode: string | null
  mtime: string
  command: string | null
}

export interface CampaignsResponse {
  campaigns: CampaignSummary[]
}

export interface CampaignLogResponse {
  name: string
  log_file: string | null
  lines: string[]
}

// ---- 报告 ----

export interface ReportFile {
  path: string
  size: number
  mtime: string
}

export interface ReportsListResponse {
  files: ReportFile[]
}

export interface ReportFileResponse {
  path: string
  size: number
  content: string
}
