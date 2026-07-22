import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Breadcrumb,
  Button,
  Drawer,
  Empty,
  Input,
  Modal,
  Popconfirm,
  Space,
  Spin,
  Table,
  Typography,
  message,
} from 'antd'
import {
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  FolderOutlined,
  FileOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import {
  deleteFile,
  fetchFileContent,
  fetchFiles,
  putFileContent,
} from '../api/client'
import type {
  FileContentResponse,
  FileDirEntry,
  FileEntry,
} from '../types'

function fmtSize(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(2)} MB`
}

function fmtMtime(iso: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString('zh-CN')
  } catch {
    return iso
  }
}

function joinPath(base: string, name: string): string {
  if (!base) return name
  return `${base.replace(/\/+$/, '')}/${name}`
}

function parentPath(path: string): string {
  if (!path) return ''
  const parts = path.split('/').filter(Boolean)
  parts.pop()
  return parts.join('/')
}

type RowItem =
  | { kind: 'dir'; name: string; mtime: string; size?: never }
  | { kind: 'file'; name: string; mtime: string; size: number }

function isTextName(name: string): boolean {
  const lower = name.toLowerCase()
  if (!lower.includes('.')) return true
  return /\.(json|jsonl|md|txt|yaml|yml|py|sh|csv|html|log|cfg|toml)$/.test(
    lower,
  )
}

function prettyText(path: string, content: string): string {
  if (path.toLowerCase().endsWith('.json')) {
    try {
      return JSON.stringify(JSON.parse(content), null, 2)
    } catch {
      return content
    }
  }
  return content
}

export function FilesPage() {
  const [path, setPath] = useState('')
  const [dirs, setDirs] = useState<FileDirEntry[]>([])
  const [files, setFiles] = useState<FileEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // 查看
  const [viewOpen, setViewOpen] = useState(false)
  const [viewPath, setViewPath] = useState('')
  const [viewData, setViewData] = useState<FileContentResponse | null>(null)
  const [viewLoading, setViewLoading] = useState(false)

  // 编辑
  const [editOpen, setEditOpen] = useState(false)
  const [editPath, setEditPath] = useState('')
  const [editContent, setEditContent] = useState('')
  const [editSaving, setEditSaving] = useState(false)

  // 非空目录删除确认
  const [rmDirOpen, setRmDirOpen] = useState(false)
  const [rmDirName, setRmDirName] = useState('')
  const [rmDirPath, setRmDirPath] = useState('')
  const [rmDirConfirm, setRmDirConfirm] = useState('')
  const [rmDirLoading, setRmDirLoading] = useState(false)

  const load = useCallback((p: string) => {
    setLoading(true)
    fetchFiles(p)
      .then((res) => {
        setDirs(res.dirs)
        setFiles(res.files)
        setPath(res.path)
        setError(null)
      })
      .catch((e: Error) => {
        setError(e.message)
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    load('')
  }, [load])

  const crumbs = useMemo(() => {
    const parts = path ? path.split('/').filter(Boolean) : []
    const items: { title: ReactNode; key: string }[] = [
      {
        key: '',
        title: (
          <a
            onClick={(e) => {
              e.preventDefault()
              load('')
            }}
          >
            workspace
          </a>
        ),
      },
    ]
    let acc = ''
    for (const part of parts) {
      acc = acc ? `${acc}/${part}` : part
      const target = acc
      items.push({
        key: target,
        title: (
          <a
            onClick={(e) => {
              e.preventDefault()
              load(target)
            }}
          >
            {part}
          </a>
        ),
      })
    }
    return items
  }, [path, load])

  const rows: RowItem[] = useMemo(() => {
    const d: RowItem[] = dirs.map((x) => ({
      kind: 'dir',
      name: x.name,
      mtime: x.mtime,
    }))
    const f: RowItem[] = files.map((x) => ({
      kind: 'file',
      name: x.name,
      mtime: x.mtime,
      size: x.size,
    }))
    return [...d, ...f]
  }, [dirs, files])

  const openView = (filePath: string) => {
    setViewPath(filePath)
    setViewData(null)
    setViewOpen(true)
    setViewLoading(true)
    fetchFileContent(filePath)
      .then((data) => setViewData(data))
      .catch((e: Error) => {
        message.error(e.message)
        setViewOpen(false)
      })
      .finally(() => setViewLoading(false))
  }

  const openEdit = (filePath: string) => {
    setEditPath(filePath)
    setEditContent('')
    setEditOpen(true)
    fetchFileContent(filePath)
      .then((data) => {
        if (data.kind === 'text') {
          setEditContent(data.content)
        } else {
          message.warning('仅支持编辑文本文件')
          setEditOpen(false)
        }
      })
      .catch((e: Error) => {
        message.error(e.message)
        setEditOpen(false)
      })
  }

  const saveEdit = async () => {
    setEditSaving(true)
    try {
      await putFileContent(editPath, editContent)
      message.success('保存成功')
      setEditOpen(false)
      load(path)
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setEditSaving(false)
    }
  }

  const doDelete = async (target: string, recursive = false) => {
    try {
      await deleteFile(target, recursive)
      message.success(`已删除 ${target}`)
      load(path)
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      // 非空目录 409 → 打开确认 Modal
      if (msg.includes('409') || msg.includes('recursive')) {
        const name = target.split('/').pop() ?? target
        setRmDirPath(target)
        setRmDirName(name)
        setRmDirConfirm('')
        setRmDirOpen(true)
        return
      }
      message.error(msg)
    }
  }

  const confirmRmDir = async () => {
    if (rmDirConfirm !== rmDirName) {
      message.warning('请输入目录名原文以确认')
      return
    }
    setRmDirLoading(true)
    try {
      await deleteFile(rmDirPath, true)
      message.success(`已递归删除 ${rmDirPath}`)
      setRmDirOpen(false)
      load(path)
    } catch (e) {
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setRmDirLoading(false)
    }
  }

  const columns: ColumnsType<RowItem> = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (_: string, record) =>
        record.kind === 'dir' ? (
          <Typography.Link
            onClick={() => load(joinPath(path, record.name))}
          >
            <FolderOutlined style={{ marginRight: 6, color: '#faad14' }} />
            {record.name}
          </Typography.Link>
        ) : (
          <span>
            <FileOutlined style={{ marginRight: 6, color: '#8c8c8c' }} />
            <Typography.Text
              style={{
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                fontSize: 13,
              }}
            >
              {record.name}
            </Typography.Text>
          </span>
        ),
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      width: 100,
      render: (_: unknown, record) =>
        record.kind === 'file' ? fmtSize(record.size) : '—',
    },
    {
      title: '修改时间',
      dataIndex: 'mtime',
      key: 'mtime',
      width: 180,
      render: (v: string) => fmtMtime(v),
    },
    {
      title: '操作',
      key: 'actions',
      width: 220,
      render: (_: unknown, record) => {
        const target = joinPath(path, record.name)
        if (record.kind === 'dir') {
          return (
            <Popconfirm
              title="删除空目录？"
              description="若目录非空将要求二次确认"
              onConfirm={() => void doDelete(target, false)}
              okText="删除"
              cancelText="取消"
            >
              <Button size="small" danger icon={<DeleteOutlined />}>
                删除
              </Button>
            </Popconfirm>
          )
        }
        const canEdit = isTextName(record.name)
        return (
          <Space size="small">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => openView(target)}
            >
              查看
            </Button>
            {canEdit && (
              <Button
                size="small"
                icon={<EditOutlined />}
                onClick={() => openEdit(target)}
              >
                编辑
              </Button>
            )}
            <Popconfirm
              title="确认删除此文件？"
              onConfirm={() => void doDelete(target, false)}
              okText="删除"
              cancelText="取消"
            >
              <Button size="small" danger icon={<DeleteOutlined />}>
                删除
              </Button>
            </Popconfirm>
          </Space>
        )
      },
    },
  ]

  const renderViewBody = () => {
    if (viewLoading) return <Spin tip="加载内容…" />
    if (!viewData) return <Empty description="无内容" />

    if (viewData.kind === 'text') {
      return (
        <pre
          style={{
            margin: 0,
            maxHeight: 'calc(100vh - 140px)',
            overflow: 'auto',
            fontSize: 12,
            background: '#fafafa',
            padding: 12,
            borderRadius: 4,
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {prettyText(viewData.path, viewData.content)}
        </pre>
      )
    }

    if (viewData.kind === 'parquet') {
      const cols = viewData.schema.map((c) => ({
        title: `${c.name} (${c.dtype})`,
        dataIndex: c.name,
        key: c.name,
        ellipsis: true,
        render: (v: unknown) =>
          v === null || v === undefined ? (
            <Typography.Text type="secondary">null</Typography.Text>
          ) : (
            String(v)
          ),
      }))
      return (
        <div>
          <Typography.Paragraph type="secondary">
            共 {viewData.n_rows} 行 · 预览前 {viewData.head.length} 行
          </Typography.Paragraph>
          <Typography.Paragraph>
            <Typography.Text strong>Schema：</Typography.Text>
            {viewData.schema.map((c) => `${c.name}:${c.dtype}`).join(', ')}
          </Typography.Paragraph>
          <Table
            size="small"
            rowKey={(_, i) => String(i)}
            columns={cols}
            dataSource={viewData.head}
            scroll={{ x: true }}
            pagination={false}
          />
        </div>
      )
    }

    return (
      <Empty description="二进制文件，仅支持删除" />
    )
  }

  if (loading && dirs.length === 0 && files.length === 0 && !error) {
    return (
      <div style={{ textAlign: 'center', padding: 48 }}>
        <Spin tip="加载目录…" />
      </div>
    )
  }

  if (error && dirs.length === 0 && files.length === 0) {
    return <Empty description={`加载失败: ${error}`} />
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        文件管理
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        当前路径：
        <Typography.Text code>{path || '/'}</Typography.Text>
        {path ? (
          <>
            {' · '}
            <Typography.Link onClick={() => load(parentPath(path))}>
              返回上级
            </Typography.Link>
          </>
        ) : null}
      </Typography.Paragraph>

      <Breadcrumb items={crumbs} style={{ marginBottom: 16 }} />

      <Table
        rowKey={(r) => `${r.kind}-${r.name}`}
        size="middle"
        loading={loading}
        columns={columns}
        dataSource={rows}
        pagination={{ pageSize: 50, showSizeChanger: true }}
        locale={{ emptyText: '空目录' }}
      />

      {/* 查看 Drawer */}
      <Drawer
        title={viewPath || '预览'}
        width={800}
        open={viewOpen}
        onClose={() => {
          setViewOpen(false)
          setViewData(null)
        }}
        destroyOnClose
      >
        {renderViewBody()}
      </Drawer>

      {/* 编辑 Drawer */}
      <Drawer
        title={`编辑 ${editPath}`}
        width={800}
        open={editOpen}
        onClose={() => setEditOpen(false)}
        destroyOnClose
        extra={
          <Button type="primary" loading={editSaving} onClick={() => void saveEdit()}>
            保存
          </Button>
        }
      >
        <Input.TextArea
          value={editContent}
          onChange={(e) => setEditContent(e.target.value)}
          autoSize={{ minRows: 16, maxRows: 40 }}
          style={{
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 13,
          }}
        />
      </Drawer>

      {/* 非空目录删除确认 */}
      <Modal
        title="递归删除非空目录"
        open={rmDirOpen}
        onCancel={() => setRmDirOpen(false)}
        onOk={() => void confirmRmDir()}
        okText="确认删除"
        okButtonProps={{
          danger: true,
          disabled: rmDirConfirm !== rmDirName,
          loading: rmDirLoading,
        }}
        cancelText="取消"
      >
        <Typography.Paragraph>
          目录 <Typography.Text code>{rmDirPath}</Typography.Text> 非空。
          请输入目录名 <Typography.Text strong>{rmDirName}</Typography.Text>{' '}
          以确认递归删除：
        </Typography.Paragraph>
        <Input
          value={rmDirConfirm}
          onChange={(e) => setRmDirConfirm(e.target.value)}
          placeholder={rmDirName}
        />
      </Modal>
    </div>
  )
}
