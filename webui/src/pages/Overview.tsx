import { useEffect, useMemo, useState } from 'react'
import { Card, Col, Empty, Row, Spin, Tag, Typography } from 'antd'
import { useNavigate } from 'react-router-dom'
import { fetchOverview } from '../api/client'
import {
  domainDesc,
  domainLabel,
  groupDomains,
} from '../domainMeta'
import type { DomainOverview } from '../types'

export function OverviewPage() {
  const navigate = useNavigate()
  const [domains, setDomains] = useState<DomainOverview[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchOverview()
      .then((res) => {
        if (!cancelled) {
          setDomains(res.domains)
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
  }, [])

  const byKey = useMemo(() => {
    const m = new Map<string, DomainOverview>()
    for (const d of domains) m.set(d.domain, d)
    return m
  }, [domains])

  const sections = useMemo(
    () => groupDomains(domains.map((d) => d.domain)),
    [domains],
  )

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="加载总览…" />
      </div>
    )
  }

  if (error) {
    return <Empty description={`加载失败: ${error}`} />
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        总览
      </Typography.Title>

      {sections.map(({ group, keys }) => (
        <div key={group} style={{ marginBottom: 28 }}>
          <Typography.Title level={5} style={{ marginBottom: 12 }}>
            {group}
          </Typography.Title>
          <Row gutter={[16, 16]}>
            {keys.map((key) => {
              const d = byKey.get(key)
              if (!d) return null
              const label = domainLabel(key)
              const desc = domainDesc(key)
              return (
                <Col key={key} xs={24} sm={12} md={8} lg={6}>
                  <Card
                    hoverable
                    size="small"
                    title={
                      <span>
                        {label}
                        <Typography.Text
                          type="secondary"
                          style={{
                            display: 'block',
                            fontSize: 12,
                            fontWeight: 400,
                            marginTop: 2,
                            whiteSpace: 'normal',
                            lineHeight: 1.4,
                          }}
                        >
                          {desc || key}
                        </Typography.Text>
                      </span>
                    }
                    onClick={() => navigate(`/domain/${key}`)}
                    extra={
                      <Tag color={d.count > 0 ? 'blue' : 'default'}>
                        {d.count}
                      </Tag>
                    }
                  >
                    {d.latest ? (
                      <div style={{ fontSize: 13 }}>
                        <div>
                          <Typography.Text type="secondary">最新 </Typography.Text>
                          <Typography.Text code>{d.latest.run_id}</Typography.Text>
                        </div>
                        <div style={{ marginTop: 4 }}>
                          <Typography.Text type="secondary">状态 </Typography.Text>
                          {d.latest.status ?? '—'}
                        </div>
                      </div>
                    ) : (
                      <Typography.Text type="secondary">暂无产物</Typography.Text>
                    )}
                  </Card>
                </Col>
              )
            })}
          </Row>
        </div>
      ))}
    </div>
  )
}
