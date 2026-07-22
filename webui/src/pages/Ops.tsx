import { useCallback, useEffect, useState } from 'react'
import {
  Button,
  Drawer,
  Empty,
  Radio,
  Space,
  Spin,
  Table,
  Tag,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { ReloadOutlined } from '@ant-design/icons'
import { fetchCampaignLog, fetchCampaigns } from '../api/client'
import type { CampaignSummary } from '../types'

function fmtMtime(iso: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('zh-CN')
  } catch {
    return iso
  }
}

export function OpsPage() {
  const [campaigns, setCampaigns] = useState<CampaignSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<CampaignSummary | null>(null)
  const [logLines, setLogLines] = useState<string[]>([])
  const [logFile, setLogFile] = useState<string | null>(null)
  const [logLoading, setLogLoading] = useState(false)
  const [tail, setTail] = useState(200)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchCampaigns()
      .then((res) => {
        if (!cancelled) {
          setCampaigns(res.campaigns)
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

  const loadLog = useCallback(
    (name: string, n: number) => {
      setLogLoading(true)
      fetchCampaignLog(name, n)
        .then((res) => {
          setLogLines(res.lines)
          setLogFile(res.log_file)
        })
        .catch(() => {
          setLogLines([])
          setLogFile(null)
        })
        .finally(() => setLogLoading(false))
    },
    [],
  )

  useEffect(() => {
    if (selected) {
      loadLog(selected.name, tail)
    }
  }, [selected, tail, loadLog])

  const columns: ColumnsType<CampaignSummary> = [
    {
      title: 'name',
      dataIndex: 'name',
      key: 'name',
      width: 220,
      render: (v: string) => <Typography.Text code>{v}</Typography.Text>,
    },
    {
      title: '状态',
      dataIndex: 'done',
      key: 'done',
      width: 100,
      render: (done: boolean) =>
        done ? <Tag color="green">done</Tag> : <Tag color="processing">running</Tag>,
    },
    {
      title: 'exitcode',
      dataIndex: 'exitcode',
      key: 'exitcode',
      width: 90,
      render: (v: string | null) =>
        v == null ? '—' : (
          <Tag color={v === '0' ? 'success' : 'error'}>{v}</Tag>
        ),
    },
    {
      title: 'mtime',
      dataIndex: 'mtime',
      key: 'mtime',
      width: 180,
      render: (v: string) => fmtMtime(v),
    },
    {
      title: 'command',
      dataIndex: 'command',
      key: 'command',
      ellipsis: true,
      render: (v: string | null) => (
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
  ]

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="加载运营任务…" />
      </div>
    )
  }

  if (error) {
    return <Empty description={`加载失败: ${error}`} />
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        运营
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        campaigns · 共 {campaigns.length} 条
      </Typography.Paragraph>
      <Table
        rowKey="name"
        size="middle"
        columns={columns}
        dataSource={campaigns}
        pagination={{ pageSize: 50, showSizeChanger: true }}
        onRow={(record) => ({
          onClick: () => setSelected(record),
          style: { cursor: 'pointer' },
        })}
        locale={{ emptyText: '暂无 campaign' }}
      />

      <Drawer
        title={selected ? `Log · ${selected.name}` : 'Log'}
        width={800}
        open={!!selected}
        onClose={() => setSelected(null)}
        destroyOnClose
        extra={
          <Space>
            <Radio.Group
              size="small"
              value={tail}
              onChange={(e) => setTail(e.target.value)}
              optionType="button"
              options={[
                { label: '100', value: 100 },
                { label: '500', value: 500 },
                { label: '2000', value: 2000 },
              ]}
            />
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => selected && loadLog(selected.name, tail)}
            >
              刷新
            </Button>
          </Space>
        }
      >
        {logLoading ? (
          <Spin tip="加载 log…" />
        ) : (
          <>
            <Typography.Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
              文件: {logFile ?? '（无 log）'} · 尾部 {logLines.length} 行
            </Typography.Text>
            {logLines.length === 0 ? (
              <Empty description="无 log 内容" image={Empty.PRESENTED_IMAGE_SIMPLE} />
            ) : (
              <pre
                style={{
                  margin: 0,
                  maxHeight: 'calc(100vh - 180px)',
                  overflow: 'auto',
                  fontSize: 12,
                  background: '#1e1e1e',
                  color: '#d4d4d4',
                  padding: 12,
                  borderRadius: 4,
                  fontFamily:
                    'ui-monospace, SFMono-Regular, Menlo, monospace',
                }}
              >
                {logLines.join('\n')}
              </pre>
            )}
          </>
        )}
      </Drawer>
    </div>
  )
}
