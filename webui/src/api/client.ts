import type {
  HealthResponse,
  NavResponse,
  OverviewResponse,
  RunDetailResponse,
  RunsResponse,
} from '../types'

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`请求失败 ${res.status}: ${url}`)
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
