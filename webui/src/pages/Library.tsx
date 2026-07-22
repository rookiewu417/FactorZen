import { useEffect, useMemo, useState } from 'react'
import {
  Card,
  Col,
  Descriptions,
  Drawer,
  Empty,
  Input,
  Row,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType, TableProps } from 'antd/es/table'
import { fetchLibrary, fetchTrack } from '../api/client'
import { IcChart } from '../components/IcChart'
import type { FactorRecord, Market, TrackPoint } from '../types'

const MARKETS: Market[] = ['ashare', 'crypto', 'us', 'futures']

const STATUS_COLOR: Record<string, string> = {
  active: 'green',
  correlated: 'orange',
  rejected: 'red',
  candidate: 'blue',
  pending: 'default',
}

function fmtNum(v: unknown, digits = 4): string {
  if (v == null || v === '') return '—'
  if (typeof v === 'number' && Number.isFinite(v)) return v.toFixed(digits)
  return String(v)
}

export function LibraryPage() {
  const [market, setMarket] = useState<Market>('ashare')
  const [factors, setFactors] = useState<FactorRecord[]>([])
  const [byStatus, setByStatus] = useState<Record<string, number>>({})
  const [count, setCount] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<FactorRecord | null>(null)
  const [trackPoints, setTrackPoints] = useState<TrackPoint[]>([])
  const [trackLoading, setTrackLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchLibrary(market)
      .then((res) => {
        if (!cancelled) {
          setFactors(res.factors)
          setByStatus(res.by_status)
          setCount(res.count)
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
  }, [market])

  useEffect(() => {
    if (!selected?.expression) {
      setTrackPoints([])
      return
    }
    let cancelled = false
    setTrackLoading(true)
    fetchTrack(market, selected.expression)
      .then((res) => {
        if (!cancelled) setTrackPoints(res.points)
      })
      .catch(() => {
        if (!cancelled) setTrackPoints([])
      })
      .finally(() => {
        if (!cancelled) setTrackLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [selected, market])

  const statusFilters = useMemo(
    () =>
      Object.keys(byStatus)
        .sort()
        .map((s) => ({ text: s, value: s })),
    [byStatus],
  )

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return factors
    return factors.filter((f) =>
      String(f.expression ?? '')
        .toLowerCase()
        .includes(q),
    )
  }, [factors, search])

  const columns: ColumnsType<FactorRecord> = [
    {
      title: 'expression',
      dataIndex: 'expression',
      key: 'expression',
      ellipsis: true,
      render: (v: string | undefined) => (
        <Typography.Text
          code
          copyable={v ? { text: v } : false}
          style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}
        >
          {v ?? '—'}
        </Typography.Text>
      ),
    },
    {
      title: 'status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      filters: statusFilters,
      onFilter: (value, record) => String(record.status ?? '') === String(value),
      render: (v: string | null | undefined) => {
        const s = v ?? 'unknown'
        return <Tag color={STATUS_COLOR[s] ?? 'default'}>{s}</Tag>
      },
    },
    {
      title: 'ic_train',
      dataIndex: 'ic_train',
      key: 'ic_train',
      width: 110,
      sorter: (a, b) =>
        (Number(a.ic_train) || 0) - (Number(b.ic_train) || 0),
      render: (v) => fmtNum(v),
    },
    {
      title: 'holdout_ic',
      dataIndex: 'holdout_ic',
      key: 'holdout_ic',
      width: 110,
      sorter: (a, b) =>
        (Number(a.holdout_ic) || 0) - (Number(b.holdout_ic) || 0),
      render: (v) => fmtNum(v),
    },
    {
      title: 'dsr',
      dataIndex: 'dsr',
      key: 'dsr',
      width: 90,
      sorter: (a, b) => (Number(a.dsr) || 0) - (Number(b.dsr) || 0),
      render: (v) => fmtNum(v),
    },
    {
      title: 'turnover',
      dataIndex: 'turnover',
      key: 'turnover',
      width: 100,
      sorter: (a, b) =>
        (Number(a.turnover) || 0) - (Number(b.turnover) || 0),
      render: (v) => fmtNum(v),
    },
    {
      title: 'admission_track',
      dataIndex: 'admission_track',
      key: 'admission_track',
      width: 130,
      render: (v: string | null | undefined) => v ?? '—',
    },
  ]

  const tableProps: TableProps<FactorRecord> = {
    rowKey: (r, i) => `${r.expression ?? ''}-${i}`,
    size: 'middle',
    columns,
    dataSource: filtered,
    pagination: { pageSize: 50, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` },
    onRow: (record) => ({
      onClick: () => setSelected(record),
      style: { cursor: 'pointer' },
    }),
    locale: { emptyText: '暂无因子' },
    scroll: { x: 960 },
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        因子库
      </Typography.Title>
      <Tabs
        activeKey={market}
        onChange={(k) => {
          setMarket(k as Market)
          setSelected(null)
          setSearch('')
        }}
        items={MARKETS.map((m) => ({ key: m, label: m }))}
        style={{ marginBottom: 8 }}
      />

      {loading ? (
        <div style={{ textAlign: 'center', padding: 48 }}>
          <Spin tip="加载因子库…" />
        </div>
      ) : error ? (
        <Empty description={`加载失败: ${error}`} />
      ) : (
        <>
          <Row gutter={[12, 12]} style={{ marginBottom: 16 }}>
            <Col xs={12} sm={8} md={4}>
              <Card size="small">
                <Statistic title="合计" value={count} />
              </Card>
            </Col>
            {Object.entries(byStatus)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([st, n]) => (
                <Col key={st} xs={12} sm={8} md={4}>
                  <Card size="small">
                    <Statistic
                      title={st}
                      value={n}
                      valueStyle={{
                        color:
                          STATUS_COLOR[st] === 'green'
                            ? '#3f8600'
                            : STATUS_COLOR[st] === 'red'
                              ? '#cf1322'
                              : undefined,
                      }}
                    />
                  </Card>
                </Col>
              ))}
          </Row>

          <Input.Search
            placeholder="搜索 expression…"
            allowClear
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ marginBottom: 12, maxWidth: 420 }}
          />

          <Table {...tableProps} />
        </>
      )}

      <Drawer
        title="因子详情"
        width={640}
        open={!!selected}
        onClose={() => setSelected(null)}
        destroyOnClose
      >
        {selected && (
          <>
            <Descriptions
              size="small"
              column={1}
              bordered
              style={{ marginBottom: 16 }}
            >
              {Object.entries(selected).map(([k, v]) => (
                <Descriptions.Item key={k} label={k}>
                  {v == null ? (
                    <Typography.Text type="secondary">null</Typography.Text>
                  ) : typeof v === 'object' ? (
                    <pre
                      style={{
                        margin: 0,
                        fontSize: 12,
                        maxHeight: 120,
                        overflow: 'auto',
                      }}
                    >
                      {JSON.stringify(v, null, 2)}
                    </pre>
                  ) : (
                    <Typography.Text
                      style={
                        k === 'expression'
                          ? {
                              fontFamily:
                                'ui-monospace, SFMono-Regular, Menlo, monospace',
                              wordBreak: 'break-all',
                            }
                          : undefined
                      }
                      copyable={k === 'expression'}
                    >
                      {String(v)}
                    </Typography.Text>
                  )}
                </Descriptions.Item>
              ))}
            </Descriptions>

            <Typography.Title level={5}>向前追踪 IC</Typography.Title>
            {trackLoading ? (
              <Spin tip="加载 track…" />
            ) : trackPoints.length === 0 ? (
              <Empty description="暂无 forward-track 数据" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <IcChart points={trackPoints} />
            )}
          </>
        )}
      </Drawer>
    </div>
  )
}
