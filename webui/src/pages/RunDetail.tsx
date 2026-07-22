import { useEffect, useState } from 'react'
import {
  Card,
  Col,
  Collapse,
  Empty,
  Row,
  Spin,
  Statistic,
  Typography,
} from 'antd'
import { useParams } from 'react-router-dom'
import { fetchNav, fetchRunDetail } from '../api/client'
import { NavChart } from '../components/NavChart'
import { domainLabel } from '../domainMeta'
import type { RunDetailResponse } from '../types'

function isNumericMetric(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

export function RunDetailPage() {
  const { domain = '', runId = '' } = useParams<{
    domain: string
    runId: string
  }>()
  const [detail, setDetail] = useState<RunDetailResponse | null>(null)
  const [nav, setNav] = useState<[string, number][]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!domain || !runId) return
    let cancelled = false
    setLoading(true)
    Promise.all([fetchRunDetail(domain, runId), fetchNav(domain, runId)])
      .then(([d, n]) => {
        if (!cancelled) {
          setDetail(d)
          setNav(n.nav ?? [])
          setError(null)
        }
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [domain, runId])

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="加载详情…" />
      </div>
    )
  }

  if (error || !detail) {
    return <Empty description={`加载失败: ${error ?? '未知错误'}`} />
  }

  const metrics = detail.metrics ?? {}
  const metricEntries = Object.entries(metrics).filter(([, v]) =>
    isNumericMetric(v) || typeof v === 'string',
  )

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        {domainLabel(domain)}{' '}
        <Typography.Text
          type="secondary"
          style={{ fontSize: 14, fontWeight: 400 }}
        >
          {domain}
        </Typography.Text>
        {' / '}
        {runId}
      </Typography.Title>

      {metricEntries.length > 0 && (
        <Card size="small" title="Metrics" style={{ marginBottom: 16 }}>
          <Row gutter={[16, 16]}>
            {metricEntries.map(([k, v]) => (
              <Col key={k} xs={12} sm={8} md={6} lg={4}>
                <Statistic
                  title={k}
                  value={typeof v === 'number' ? v : String(v)}
                  precision={typeof v === 'number' ? 4 : undefined}
                />
              </Col>
            ))}
          </Row>
        </Card>
      )}

      {nav.length > 0 && (
        <Card size="small" title="NAV 曲线" style={{ marginBottom: 16 }}>
          <NavChart data={nav} />
        </Card>
      )}

      <Card size="small" title="Manifest">
        <Collapse
          items={[
            {
              key: 'manifest',
              label: '查看原始 JSON',
              children: (
                <pre
                  style={{
                    margin: 0,
                    maxHeight: 480,
                    overflow: 'auto',
                    fontSize: 12,
                    background: '#fafafa',
                    padding: 12,
                    borderRadius: 4,
                  }}
                >
                  {JSON.stringify(detail.manifest, null, 2)}
                </pre>
              ),
            },
          ]}
        />
      </Card>
    </div>
  )
}
