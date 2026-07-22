import { useEffect, useState } from 'react'
import { Empty, Spin, Table, Typography } from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { useNavigate, useParams } from 'react-router-dom'
import { fetchRuns } from '../api/client'
import type { RunSummary } from '../types'

function shortSha(sha: string | null): string {
  if (!sha) return '—'
  return sha.length > 8 ? sha.slice(0, 8) : sha
}

export function DomainListPage() {
  const { domain = '' } = useParams<{ domain: string }>()
  const navigate = useNavigate()
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!domain) return
    let cancelled = false
    setLoading(true)
    fetchRuns(domain)
      .then((res) => {
        if (!cancelled) {
          setRuns(res.runs)
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
  }, [domain])

  const columns: ColumnsType<RunSummary> = [
    {
      title: 'run_id',
      dataIndex: 'run_id',
      key: 'run_id',
      render: (v: string) => <Typography.Text code>{v}</Typography.Text>,
    },
    {
      title: 'status',
      dataIndex: 'status',
      key: 'status',
      width: 140,
      render: (v: string | null) => v ?? '—',
    },
    {
      title: 'git_sha',
      dataIndex: 'git_sha',
      key: 'git_sha',
      width: 120,
      render: (v: string | null) => shortSha(v),
    },
  ]

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="加载 runs…" />
      </div>
    )
  }

  if (error) {
    return <Empty description={`加载失败: ${error}`} />
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        {domain}
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        共 {runs.length} 条产物
      </Typography.Paragraph>
      <Table
        rowKey="run_id"
        size="middle"
        columns={columns}
        dataSource={runs}
        pagination={{ pageSize: 50, showSizeChanger: true }}
        onRow={(record) => ({
          onClick: () => navigate(`/run/${domain}/${record.run_id}`),
          style: { cursor: 'pointer' },
        })}
        locale={{ emptyText: '暂无产物' }}
      />
    </div>
  )
}
