import { useEffect, useState } from 'react'
import {
  Button,
  Drawer,
  Empty,
  Spin,
  Table,
  Typography,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import { fetchReportFile, fetchReports, fileRawUrl } from '../api/client'
import type { ReportFile } from '../types'

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

// 报告文件 path 相对 workspace/factors/reports/；raw 端点要相对 workspace 根。
function isHtml(path: string): boolean {
  return /\.html?$/i.test(path)
}

function openRaw(path: string): void {
  window.open(fileRawUrl(`factors/reports/${path}`), '_blank', 'noopener,noreferrer')
}

function fmtMtime(iso: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('zh-CN')
  } catch {
    return iso
  }
}

function previewContent(path: string, content: string): string {
  if (path.toLowerCase().endsWith('.json')) {
    try {
      return JSON.stringify(JSON.parse(content), null, 2)
    } catch {
      return content
    }
  }
  return content
}

export function ReportsPage() {
  const [files, setFiles] = useState<ReportFile[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [content, setContent] = useState<string | null>(null)
  const [contentLoading, setContentLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchReports()
      .then((res) => {
        if (!cancelled) {
          setFiles(res.files)
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

  const openFile = (path: string) => {
    setSelectedPath(path)
    setContent(null)
    setContentLoading(true)
    fetchReportFile(path)
      .then((res) => {
        setContent(previewContent(path, res.content))
      })
      .catch((e: Error) => {
        setContent(`加载失败: ${e.message}`)
      })
      .finally(() => setContentLoading(false))
  }

  const columns: ColumnsType<ReportFile> = [
    {
      title: 'path',
      dataIndex: 'path',
      key: 'path',
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
      title: 'size',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (v: number) => fmtSize(v),
    },
    {
      title: 'mtime',
      dataIndex: 'mtime',
      key: 'mtime',
      width: 180,
      render: (v: string) => fmtMtime(v),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_: unknown, record) =>
        isHtml(record.path) ? (
          <Button
            size="small"
            type="primary"
            onClick={(e) => {
              e.stopPropagation()
              openRaw(record.path)
            }}
          >
            打开报告
          </Button>
        ) : (
          <Button
            size="small"
            onClick={(e) => {
              e.stopPropagation()
              openFile(record.path)
            }}
          >
            查看
          </Button>
        ),
    },
  ]

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="加载报告列表…" />
      </div>
    )
  }

  if (error) {
    return <Empty description={`加载失败: ${error}`} />
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        报告
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        共 {files.length} 个文件
      </Typography.Paragraph>
      <Table
        rowKey="path"
        size="middle"
        columns={columns}
        dataSource={files}
        pagination={{ pageSize: 50, showSizeChanger: true }}
        onRow={(record) => ({
          onClick: () =>
            isHtml(record.path) ? openRaw(record.path) : openFile(record.path),
          style: { cursor: 'pointer' },
        })}
        locale={{ emptyText: '暂无报告' }}
      />

      <Drawer
        title={selectedPath ?? '预览'}
        width={720}
        open={!!selectedPath}
        onClose={() => {
          setSelectedPath(null)
          setContent(null)
        }}
        destroyOnClose
      >
        {contentLoading ? (
          <Spin tip="加载内容…" />
        ) : content != null ? (
          <pre
            style={{
              margin: 0,
              maxHeight: 'calc(100vh - 140px)',
              overflow: 'auto',
              fontSize: 12,
              background: '#fafafa',
              padding: 12,
              borderRadius: 4,
              fontFamily:
                'ui-monospace, SFMono-Regular, Menlo, monospace',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {content}
          </pre>
        ) : (
          <Empty description="无内容" />
        )}
      </Drawer>
    </div>
  )
}
