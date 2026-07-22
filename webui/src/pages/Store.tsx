import { useEffect, useState } from 'react'
import {
  Card,
  Col,
  Descriptions,
  Drawer,
  Empty,
  Row,
  Spin,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { fetchStore, fetchStoreDetail } from '../api/client'
import type { Market, StoreDetailResponse, StoreEntry } from '../types'

const MARKETS: Market[] = ['ashare', 'crypto', 'us', 'futures']

const STATUS_COLOR: Record<string, string> = {
  active: 'green',
  correlated: 'orange',
  rejected: 'red',
}

export function StorePage() {
  const [market, setMarket] = useState<Market>('ashare')
  const [entries, setEntries] = useState<StoreEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [detail, setDetail] = useState<StoreDetailResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchStore(market)
      .then((res) => {
        if (!cancelled) {
          setEntries(res.entries)
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

  const openDetail = (name: string) => {
    setDetailLoading(true)
    setDetail(null)
    fetchStoreDetail(market, name)
      .then((d) => setDetail(d))
      .catch(() => setDetail(null))
      .finally(() => setDetailLoading(false))
  }

  const columns: ColumnsType<StoreEntry> = [
    {
      title: 'name',
      dataIndex: 'name',
      key: 'name',
      width: 180,
      render: (v: string) => <Typography.Text code>{v}</Typography.Text>,
    },
    {
      title: 'expression',
      dataIndex: 'expression',
      key: 'expression',
      ellipsis: true,
      render: (v: string | undefined) => (
        <Typography.Text
          style={{
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 12,
          }}
        >
          {v ?? '—'}
        </Typography.Text>
      ),
    },
    {
      title: 'created_at',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 120,
      render: (v: string | undefined) => v ?? '—',
    },
    {
      title: 'status',
      key: 'ledger_status',
      width: 120,
      render: (_, r) => {
        const s = r.ledger_snapshot?.status
        if (!s) return '—'
        return <Tag color={STATUS_COLOR[s] ?? 'default'}>{s}</Tag>
      },
    },
  ]

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        因子资产
      </Typography.Title>
      <Tabs
        activeKey={market}
        onChange={(k) => {
          setMarket(k as Market)
          setDetail(null)
        }}
        items={MARKETS.map((m) => ({ key: m, label: m }))}
        style={{ marginBottom: 8 }}
      />

      {loading ? (
        <div style={{ textAlign: 'center', padding: 48 }}>
          <Spin tip="加载因子资产…" />
        </div>
      ) : error ? (
        <Empty description={`加载失败: ${error}`} />
      ) : entries.length === 0 ? (
        <Empty description="暂无资产" />
      ) : entries.length <= 24 ? (
        <Row gutter={[12, 12]}>
          {entries.map((e) => (
            <Col key={e.name} xs={24} sm={12} md={8} lg={6}>
              <Card
                size="small"
                hoverable
                title={<Typography.Text code>{e.name}</Typography.Text>}
                extra={
                  e.ledger_snapshot?.status ? (
                    <Tag
                      color={
                        STATUS_COLOR[e.ledger_snapshot.status] ?? 'default'
                      }
                    >
                      {e.ledger_snapshot.status}
                    </Tag>
                  ) : null
                }
                onClick={() => openDetail(e.name)}
              >
                <Typography.Paragraph
                  ellipsis={{ rows: 2 }}
                  style={{
                    marginBottom: 4,
                    fontFamily:
                      'ui-monospace, SFMono-Regular, Menlo, monospace',
                    fontSize: 12,
                  }}
                >
                  {e.expression ?? '—'}
                </Typography.Paragraph>
                <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                  {e.created_at ?? '—'}
                </Typography.Text>
              </Card>
            </Col>
          ))}
        </Row>
      ) : (
        <Table
          rowKey="name"
          size="middle"
          columns={columns}
          dataSource={entries}
          pagination={{ pageSize: 50, showSizeChanger: true }}
          onRow={(record) => ({
            onClick: () => openDetail(record.name),
            style: { cursor: 'pointer' },
          })}
        />
      )}

      <Drawer
        title={detail?.name ?? '资产详情'}
        width={720}
        open={detailLoading || !!detail}
        onClose={() => setDetail(null)}
        destroyOnClose
      >
        {detailLoading && !detail ? (
          <Spin tip="加载详情…" />
        ) : detail ? (
          <>
            <Descriptions
              size="small"
              column={1}
              bordered
              style={{ marginBottom: 16 }}
              title="Meta"
            >
              {Object.entries(detail.meta).map(([k, v]) => (
                <Descriptions.Item key={k} label={k}>
                  {v == null ? (
                    <Typography.Text type="secondary">null</Typography.Text>
                  ) : typeof v === 'object' ? (
                    <pre
                      style={{
                        margin: 0,
                        fontSize: 12,
                        maxHeight: 160,
                        overflow: 'auto',
                      }}
                    >
                      {JSON.stringify(v, null, 2)}
                    </pre>
                  ) : (
                    String(v)
                  )}
                </Descriptions.Item>
              ))}
            </Descriptions>

            <Typography.Title level={5}>factor.py</Typography.Title>
            {detail.source ? (
              <pre
                style={{
                  margin: 0,
                  maxHeight: 480,
                  overflow: 'auto',
                  fontSize: 12,
                  background: '#fafafa',
                  padding: 12,
                  borderRadius: 4,
                  fontFamily:
                    'ui-monospace, SFMono-Regular, Menlo, monospace',
                }}
              >
                {detail.source}
              </pre>
            ) : (
              <Empty
                description="无 factor.py"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            )}
          </>
        ) : (
          <Empty description="加载失败" />
        )}
      </Drawer>
    </div>
  )
}
