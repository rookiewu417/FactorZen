import { useEffect, useState } from 'react'
import {
  Button,
  Card,
  Col,
  Collapse,
  Drawer,
  Empty,
  Row,
  Space,
  Spin,
  Statistic,
  Table,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useParams } from 'react-router-dom'
import {
  fetchFileContent,
  fetchFiles,
  fetchNav,
  fetchRunDetail,
  fileRawUrl,
} from '../api/client'
import { NavChart } from '../components/NavChart'
import { domainLabel } from '../domainMeta'
import type { FileContentResponse, FileEntry, RunDetailResponse } from '../types'

function isNumericMetric(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function isHtml(name: string): boolean {
  return /\.html?$/i.test(name)
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

  // 产物文件
  const [artifacts, setArtifacts] = useState<FileEntry[]>([])
  const [artLoading, setArtLoading] = useState(false)

  // 查看 Drawer
  const [viewOpen, setViewOpen] = useState(false)
  const [viewPath, setViewPath] = useState('')
  const [viewData, setViewData] = useState<FileContentResponse | null>(null)
  const [viewLoading, setViewLoading] = useState(false)

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

  useEffect(() => {
    if (!domain || !runId || !detail) return
    let cancelled = false
    setArtLoading(true)
    // 真实 run 目录以 server 返回的 path 为准（factor_evaluations 为嵌套路径）
    const runPath = detail.path ?? `${domain}/${runId}`
    fetchFiles(runPath)
      .then((res) => {
        if (!cancelled) setArtifacts(res.files)
      })
      .catch(() => {
        if (!cancelled) setArtifacts([])
      })
      .finally(() => {
        if (!cancelled) setArtLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [domain, runId, detail])

  const openView = (relPath: string) => {
    setViewPath(relPath)
    setViewData(null)
    setViewOpen(true)
    setViewLoading(true)
    fetchFileContent(relPath)
      .then((data) => setViewData(data))
      .catch((e: Error) => {
        message.error(e.message)
        setViewOpen(false)
      })
      .finally(() => setViewLoading(false))
  }

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

  const runPath = detail.path ?? `${domain}/${runId}`

  const artColumns: ColumnsType<FileEntry> = [
    {
      title: '文件',
      dataIndex: 'name',
      key: 'name',
      render: (v: string) => (
        <Typography.Text
          style={{
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 13,
          }}
        >
          {v}
        </Typography.Text>
      ),
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (n: number) => fmtSize(n),
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_: unknown, record) => {
        const full = `${runPath}/${record.name}`
        if (isHtml(record.name)) {
          return (
            <Button
              size="small"
              type="primary"
              onClick={() =>
                window.open(fileRawUrl(full), '_blank', 'noopener,noreferrer')
              }
            >
              打开报告
            </Button>
          )
        }
        return (
          <Button size="small" onClick={() => openView(full)}>
            查看
          </Button>
        )
      },
    },
  ]

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

      <Card
        size="small"
        title="产物文件"
        style={{ marginBottom: 16 }}
        extra={
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>
            {runPath}/
          </Typography.Text>
        }
      >
        <Table
          rowKey="name"
          size="small"
          loading={artLoading}
          columns={artColumns}
          dataSource={artifacts}
          pagination={false}
          locale={{ emptyText: '目录为空或不可读' }}
        />
      </Card>

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

      <Drawer
        title={viewPath || '预览'}
        width={720}
        open={viewOpen}
        onClose={() => {
          setViewOpen(false)
          setViewData(null)
        }}
        destroyOnClose
      >
        {viewLoading ? (
          <Spin tip="加载内容…" />
        ) : !viewData ? (
          <Empty description="无内容" />
        ) : viewData.kind === 'text' ? (
          <pre
            style={{
              margin: 0,
              maxHeight: 'calc(100vh - 140px)',
              overflow: 'auto',
              fontSize: 12,
              background: '#fafafa',
              padding: 12,
              borderRadius: 4,
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {viewData.content}
          </pre>
        ) : viewData.kind === 'parquet' ? (
          <Typography.Paragraph type="secondary">
            parquet · {viewData.n_rows} 行（预览见文件管理）
          </Typography.Paragraph>
        ) : (
          <Space direction="vertical">
            <Empty description="二进制文件" />
            <Button
              onClick={() =>
                window.open(fileRawUrl(viewPath), '_blank', 'noopener,noreferrer')
              }
            >
              尝试新标签打开
            </Button>
          </Space>
        )}
      </Drawer>
    </div>
  )
}
