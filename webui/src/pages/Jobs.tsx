import { useCallback, useEffect, useState } from 'react'
import {
  Button,
  Drawer,
  Popconfirm,
  Space,
  Switch,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  fetchJob,
  fetchJobLog,
  fetchJobs,
  killJob,
} from '../api/client'
import type { JobSummary } from '../types'

function statusTag(job: JobSummary) {
  if (job.status === 'running') {
    return <Tag color="blue">running</Tag>
  }
  if (job.status === 'orphaned') {
    return <Tag color="default">orphaned</Tag>
  }
  // finished
  const code = job.exit_code
  if (code === 0) {
    return <Tag color="green">finished (0)</Tag>
  }
  return <Tag color="red">finished ({code ?? '?'})</Tag>
}

function fmtTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('zh-CN')
  } catch {
    return iso
  }
}

export function JobsPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)

  // Drawer
  const [open, setOpen] = useState(false)
  const [detail, setDetail] = useState<JobSummary | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])
  const [logAuto, setLogAuto] = useState(true)
  const [killing, setKilling] = useState(false)

  const loadList = useCallback(() => {
    return fetchJobs()
      .then((res) => {
        setJobs(res.jobs)
      })
      .catch((e: Error) => {
        message.error(e.message)
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    void loadList()
  }, [loadList])

  useEffect(() => {
    if (!autoRefresh) return
    const t = window.setInterval(() => {
      void loadList()
    }, 3000)
    return () => window.clearInterval(t)
  }, [autoRefresh, loadList])

  const loadDetail = useCallback((jobId: string) => {
    return Promise.all([fetchJob(jobId), fetchJobLog(jobId, 500)])
      .then(([d, log]) => {
        setDetail(d)
        setLogLines(log.lines)
      })
      .catch((e: Error) => {
        message.error(e.message)
      })
  }, [])

  useEffect(() => {
    if (!open || !detail || !logAuto) return
    const t = window.setInterval(() => {
      void loadDetail(detail.job_id)
    }, 3000)
    return () => window.clearInterval(t)
  }, [open, detail, logAuto, loadDetail])

  const openDrawer = (job: JobSummary) => {
    setDetail(job)
    setLogLines([])
    setOpen(true)
    setLogAuto(true)
    void loadDetail(job.job_id)
  }

  const doKill = async () => {
    if (!detail) return
    setKilling(true)
    try {
      await killJob(detail.job_id)
      message.success('已发送终止信号')
      await loadDetail(detail.job_id)
      await loadList()
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setKilling(false)
    }
  }

  const columns: ColumnsType<JobSummary> = [
    {
      title: 'title',
      dataIndex: 'title',
      key: 'title',
      ellipsis: true,
    },
    {
      title: 'kind',
      dataIndex: 'kind',
      key: 'kind',
      width: 80,
      render: (v: string) => <Typography.Text code>{v}</Typography.Text>,
    },
    {
      title: 'status',
      key: 'status',
      width: 140,
      render: (_: unknown, record) => statusTag(record),
    },
    {
      title: 'started_at',
      dataIndex: 'started_at',
      key: 'started_at',
      width: 180,
      render: (v: string) => fmtTime(v),
    },
    {
      title: 'job_id',
      dataIndex: 'job_id',
      key: 'job_id',
      width: 200,
      render: (v: string) => (
        <Typography.Text
          code
          style={{ fontSize: 12 }}
          copyable={{ text: v }}
        >
          {v}
        </Typography.Text>
      ),
    },
  ]

  return (
    <div>
      <Space
        style={{
          width: '100%',
          justifyContent: 'space-between',
          marginBottom: 16,
        }}
      >
        <Typography.Title level={4} style={{ margin: 0 }}>
          任务中心
        </Typography.Title>
        <Space>
          <Typography.Text type="secondary">自动刷新</Typography.Text>
          <Switch checked={autoRefresh} onChange={setAutoRefresh} />
        </Space>
      </Space>

      <Table
        rowKey="job_id"
        size="middle"
        loading={loading}
        columns={columns}
        dataSource={jobs}
        pagination={{ pageSize: 30, showSizeChanger: true }}
        onRow={(record) => ({
          onClick: () => openDrawer(record),
          style: { cursor: 'pointer' },
        })}
        locale={{ emptyText: '暂无任务' }}
      />

      <Drawer
        title={detail?.title ?? '任务详情'}
        width={720}
        open={open}
        onClose={() => {
          setOpen(false)
          setDetail(null)
        }}
        destroyOnClose
        extra={
          detail?.status === 'running' ? (
            <Popconfirm
              title="确认终止此任务？"
              onConfirm={() => void doKill()}
              okText="终止"
              cancelText="取消"
            >
              <Button danger loading={killing}>
                终止
              </Button>
            </Popconfirm>
          ) : null
        }
      >
        {detail && (
          <>
            <Typography.Paragraph>
              <Typography.Text type="secondary">job_id：</Typography.Text>
              <Typography.Text code>{detail.job_id}</Typography.Text>
              {' · '}
              {statusTag(detail)}
              {' · '}
              <Typography.Text type="secondary">kind=</Typography.Text>
              {detail.kind}
            </Typography.Paragraph>
            <Typography.Paragraph>
              <Typography.Text strong>argv</Typography.Text>
              <pre
                style={{
                  marginTop: 8,
                  background: '#fafafa',
                  padding: 12,
                  borderRadius: 4,
                  fontSize: 12,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-all',
                }}
              >
                {JSON.stringify(detail.argv, null, 2)}
              </pre>
            </Typography.Paragraph>
            <Space style={{ marginBottom: 8 }}>
              <Typography.Text strong>日志尾部</Typography.Text>
              <Typography.Text type="secondary">自动刷新</Typography.Text>
              <Switch checked={logAuto} onChange={setLogAuto} size="small" />
            </Space>
            <pre
              style={{
                margin: 0,
                maxHeight: 'calc(100vh - 360px)',
                overflow: 'auto',
                fontSize: 12,
                background: '#1e1e1e',
                color: '#d4d4d4',
                padding: 12,
                borderRadius: 4,
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}
            >
              {logLines.length ? logLines.join('\n') : '（暂无日志）'}
            </pre>
          </>
        )}
      </Drawer>
    </div>
  )
}
