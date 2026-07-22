import type {
  CampaignLogResponse,
  CampaignsResponse,
  FileContentResponse,
  FileDeleteResponse,
  FileWriteResponse,
  FilesListResponse,
  HealthResponse,
  LibraryResponse,
  NavResponse,
  OverviewResponse,
  ReportFileResponse,
  ReportsListResponse,
  RunDetailResponse,
  RunsResponse,
  StoreDetailResponse,
  StoreListResponse,
  TrackResponse,
} from '../types'

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`请求失败 ${res.status}: ${url}`)
  }
  return res.json() as Promise<T>
}

async function requestJson<T>(
  url: string,
  init: RequestInit,
): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = `${res.status}`
    try {
      const body = (await res.json()) as { detail?: string }
      if (body.detail) detail = `${res.status}: ${body.detail}`
    } catch {
      // ignore
    }
    throw new Error(`请求失败 ${detail}: ${url}`)
  }
  return res.json() as Promise<T>
}

export function fetchHealth(): Promise<HealthResponse> {
  return getJson('/api/health')
}

export function fetchOverview(): Promise<OverviewResponse> {
  return getJson('/api/overview')
}

export function fetchRuns(domain: string): Promise<RunsResponse> {
  return getJson(`/api/runs?domain=${encodeURIComponent(domain)}`)
}

export function fetchRunDetail(
  domain: string,
  runId: string,
): Promise<RunDetailResponse> {
  return getJson(
    `/api/runs/${encodeURIComponent(domain)}/${encodeURIComponent(runId)}`,
  )
}

export function fetchNav(domain: string, runId: string): Promise<NavResponse> {
  return getJson(
    `/api/nav/${encodeURIComponent(domain)}/${encodeURIComponent(runId)}`,
  )
}

// ---- 因子库 ----

export function fetchLibrary(market: string): Promise<LibraryResponse> {
  return getJson(`/api/library/${encodeURIComponent(market)}`)
}

export function fetchTrack(
  market: string,
  expression: string,
): Promise<TrackResponse> {
  return getJson(
    `/api/library/${encodeURIComponent(market)}/track?expression=${encodeURIComponent(expression)}`,
  )
}

// ---- 因子资产 ----

export function fetchStore(market: string): Promise<StoreListResponse> {
  return getJson(`/api/store/${encodeURIComponent(market)}`)
}

export function fetchStoreDetail(
  market: string,
  name: string,
): Promise<StoreDetailResponse> {
  return getJson(
    `/api/store/${encodeURIComponent(market)}/${encodeURIComponent(name)}`,
  )
}

// ---- 运营 ----

export function fetchCampaigns(): Promise<CampaignsResponse> {
  return getJson('/api/ops/campaigns')
}

export function fetchCampaignLog(
  name: string,
  tail = 200,
): Promise<CampaignLogResponse> {
  return getJson(
    `/api/ops/campaigns/${encodeURIComponent(name)}/log?tail=${tail}`,
  )
}

// ---- 报告 ----

export function fetchReports(): Promise<ReportsListResponse> {
  return getJson('/api/reports')
}

export function fetchReportFile(path: string): Promise<ReportFileResponse> {
  return getJson(`/api/reports/file?path=${encodeURIComponent(path)}`)
}

// ---- 文件管理 ----

export function fetchFiles(path = ''): Promise<FilesListResponse> {
  return getJson(`/api/files?path=${encodeURIComponent(path)}`)
}

export function fetchFileContent(path: string): Promise<FileContentResponse> {
  return getJson(`/api/files/content?path=${encodeURIComponent(path)}`)
}

export function putFileContent(
  path: string,
  content: string,
): Promise<FileWriteResponse> {
  return requestJson('/api/files/content', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, content }),
  })
}

export function deleteFile(
  path: string,
  recursive = false,
): Promise<FileDeleteResponse> {
  return requestJson(
    `/api/files?path=${encodeURIComponent(path)}&recursive=${recursive}`,
    { method: 'DELETE' },
  )
}
