import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card,
  Col,
  Descriptions,
  Drawer,
  Empty,
  Input,
  message,
  Popconfirm,
  Row,
  Select,
  Spin,
  Statistic,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType, TableProps } from 'antd/es/table'
import { fetchLibrary, fetchTrack, updateFactorStatus } from '../api/client'
import { IcChart } from '../components/IcChart'
import type {
  FactorRecord,
  FactorSource,
  FactorStatus,
  Market,
  TrackPoint,
} from '../types'

const MARKETS: Market[] = ['ashare', 'crypto', 'us', 'futures']

const STATUS_OPTIONS: FactorStatus[] = [
  'active',
  'correlated',
  'probation',
  'no_lift',
  'manual',
]

const STATUS_COLOR: Record<string, string> = {
  active: 'green',
  correlated: 'orange',
  probation: 'gold',
  no_lift: 'default',
  manual: 'blue',
}

/** 改为 active/probation 会进入物化与组合优化，需二次确认 */
const GUARDRAIL_STATUSES = new Set(['active', 'probation'])

function fmtNum(v: unknown, digits = 4): string {
  if (v == null || v === '') return '—'
  if (typeof v === 'number' && Number.isFinite(v)) return v.toFixed(digits)
  return String(v)
}

function sourceLabel(source: string | undefined): string {
  return source === 'store' ? '手写' : '挖掘'
}

function sourceColor(source: string | undefined): string {
  return source === 'store' ? 'blue' : 'default'
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
  const [updatingKey, setUpdatingKey] = useState<string | null>(null)
  // 待确认的「高风险」状态切换（active/probation）
  const [pendingChange, setPendingChange] = useState<{
    record: FactorRecord
    next: string
  } | null>(null)

  const reload = useCallback(() => {
    setLoading(true)
    return fetchLibrary(market)
      .then((res) => {
        setFactors(res.factors)
        setByStatus(res.by_status)
        setCount(res.count)
        setError(null)
        // 同步刷新已打开的 Drawer
        setSelected((prev) => {
          if (!prev?.expression) return prev
          const next = res.factors.find(
            (f) =>
              f.expression === prev.expression &&
              (f.source ?? 'library') === (prev.source ?? 'library'),
          )
          return next ?? prev
        })
      })
      .catch((e: Error) => {
        setError(e.message)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [market])

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

  const applyStatus = useCallback(
    async (record: FactorRecord, nextStatus: string) => {
      const expr = record.expression
      if (!expr) {
        message.error('缺少 expression，无法改状态')
        return
      }
      const source = (record.source ?? 'library') as FactorSource
      const key = `${source}:${expr}`
      setUpdatingKey(key)
      try {
        await updateFactorStatus(market, expr, nextStatus, source)
        message.success(`已更新 status → ${nextStatus}`)
        await reload()
      } catch (e) {
        message.error(e instanceof Error ? e.message : String(e))
      } finally {
        setUpdatingKey(null)
        setPendingChange(null)
      }
    },
    [market, reload],
  )

  const onStatusSelect = useCallback(
    (record: FactorRecord, nextStatus: string) => {
      const cur = String(record.status ?? '')
      if (nextStatus === cur) return
      if (GUARDRAIL_STATUSES.has(nextStatus)) {
        setPendingChange({ record, next: nextStatus })
        return
      }
      void applyStatus(record, nextStatus)
    },
    [applyStatus],
  )

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

  const renderStatusSelect = (record: FactorRecord) => {
    const s = String(record.status ?? 'unknown')
    const expr = record.expression ?? ''
    const source = (record.source ?? 'library') as string
    const key = `${source}:${expr}`
    const busy = updatingKey === key
    return (
      <Select
        size="small"
        value={STATUS_OPTIONS.includes(s as FactorStatus) ? s : s}
        style={{ width: 118 }}
        loading={busy}
        disabled={busy || !expr}
        options={STATUS_OPTIONS.map((st) => ({
          value: st,
          label: (
            <Tag
              color={STATUS_COLOR[st] ?? 'default'}
              style={{ marginInlineEnd: 0 }}
            >
              {st}
            </Tag>
          ),
        }))}
        onClick={(e) => e.stopPropagation()}
        onChange={(v) => {
          onStatusSelect(record, String(v))
        }}
      />
    )
  }

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
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 80,
      filters: [
        { text: '挖掘', value: 'library' },
        { text: '手写', value: 'store' },
      ],
      onFilter: (value, record) =>
        String(record.source ?? 'library') === String(value),
      render: (v: string | undefined) => (
        <Tag color={sourceColor(v)}>{sourceLabel(v)}</Tag>
      ),
    },
    {
      title: 'status',
      dataIndex: 'status',
      key: 'status',
      width: 140,
      filters: statusFilters,
      onFilter: (value, record) => String(record.status ?? '') === String(value),
      render: (_v, record) => renderStatusSelect(record),
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
    rowKey: (r, i) => `${r.source ?? 'library'}-${r.expression ?? ''}-${i}`,
    size: 'middle',
    columns,
    dataSource: filtered,
    pagination: { pageSize: 50, showSizeChanger: true, showTotal: (t) => `共 ${t} 条` },
    onRow: (record) => ({
      onClick: () => setSelected(record),
      style: { cursor: 'pointer' },
    }),
    locale: { emptyText: '暂无因子' },
    scroll: { x: 1040 },
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
          setPendingChange(null)
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
                            : STATUS_COLOR[st] === 'blue'
                              ? '#1677ff'
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
            <div style={{ marginBottom: 16 }}>
              <Typography.Text type="secondary" style={{ marginRight: 8 }}>
                改状态
              </Typography.Text>
              {renderStatusSelect(selected)}
              <Tag
                color={sourceColor(selected.source as string | undefined)}
                style={{ marginLeft: 12 }}
              >
                {sourceLabel(selected.source as string | undefined)}
              </Tag>
            </div>

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

      {/* 改成 active/probation 的护栏确认 */}
      <Popconfirm
        title="确认修改状态？"
        description="会让该因子进入物化与组合优化，覆盖护栏裁决"
        open={!!pendingChange}
        onConfirm={() => {
          if (pendingChange) {
            void applyStatus(pendingChange.record, pendingChange.next)
          }
        }}
        onCancel={() => setPendingChange(null)}
        okText="确认"
        cancelText="取消"
        // 锚定到页面中心附近（无具体 DOM 触发时仍可用）
        placement="top"
      >
        {/* 占位，保证 Popconfirm 有 mount 节点 */}
        <span style={{ position: 'fixed', bottom: 24, right: 24, width: 0, height: 0 }} />
      </Popconfirm>
    </div>
  )
}
