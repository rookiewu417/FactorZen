import { useEffect, useMemo, useState } from 'react'
import {
  Button,
  Card,
  Col,
  Empty,
  Form,
  Input,
  Row,
  Select,
  Space,
  Switch,
  Tree,
  Typography,
  message,
} from 'antd'
import type { DataNode } from 'antd/es/tree'
import { Link } from 'react-router-dom'
import { fetchCliSchema, submitJob } from '../api/client'
import type { CliNode, CliOption } from '../types'

function pickFlag(opt: CliOption): string {
  const long = opt.flags.find((f) => f.startsWith('--'))
  return long ?? opt.flags[0] ?? opt.dest
}

function nodeKey(path: string[]): string {
  return path.join('/')
}

function toTreeData(node: CliNode, path: string[] = []): DataNode[] {
  // 根节点 fz 的 children 作为顶层
  if (path.length === 0 && node.name === 'fz') {
    return node.children.map((c) => toTreeData(c, [c.name])[0])
  }
  const key = nodeKey(path)
  const isLeaf = !node.children || node.children.length === 0
  return [
    {
      key,
      title: (
        <span>
          <Typography.Text code>{node.name}</Typography.Text>
          {node.help ? (
            <Typography.Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
              {node.help}
            </Typography.Text>
          ) : null}
        </span>
      ),
      isLeaf,
      children: isLeaf
        ? undefined
        : node.children.map((c) => toTreeData(c, [...path, c.name])[0]),
    },
  ]
}

function findNode(root: CliNode, path: string[]): CliNode | null {
  let cur: CliNode = root
  // 若 path 相对 fz 子树
  const start = root.name === 'fz' ? root : root
  cur = start
  for (const seg of path) {
    const next = cur.children.find((c) => c.name === seg)
    if (!next) return null
    cur = next
  }
  return cur
}

export function CliPage() {
  const [schema, setSchema] = useState<CliNode | null>(null)
  const [loading, setLoading] = useState(true)
  const [selectedPath, setSelectedPath] = useState<string[]>([])
  const [form] = Form.useForm()
  const [extra, setExtra] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [lastJobId, setLastJobId] = useState<string | null>(null)

  useEffect(() => {
    fetchCliSchema()
      .then(setSchema)
      .catch((e: Error) => message.error(e.message))
      .finally(() => setLoading(false))
  }, [])

  const treeData = useMemo(
    () => (schema ? toTreeData(schema) : []),
    [schema],
  )

  const selectedNode = useMemo(() => {
    if (!schema || selectedPath.length === 0) return null
    return findNode(schema, selectedPath)
  }, [schema, selectedPath])

  const isLeaf =
    selectedNode != null &&
    (!selectedNode.children || selectedNode.children.length === 0)

  useEffect(() => {
    form.resetFields()
  }, [selectedPath, form])

  const buildArgv = (values: Record<string, unknown>): string[] => {
    if (!selectedNode) return []
    const argv: string[] = [...selectedPath]
    const positionals = selectedNode.options.filter((o) => o.is_positional)
    const options = selectedNode.options.filter((o) => !o.is_positional)

    for (const opt of positionals) {
      const v = values[opt.dest]
      if (v === undefined || v === null || v === '') {
        if (opt.required) {
          throw new Error(`缺少必填参数: ${opt.dest}`)
        }
        continue
      }
      argv.push(String(v))
    }

    for (const opt of options) {
      const v = values[opt.dest]
      if (opt.is_flag) {
        if (v === true) {
          argv.push(pickFlag(opt))
        }
        continue
      }
      if (v === undefined || v === null || v === '') continue
      argv.push(pickFlag(opt), String(v))
    }

    const extras = extra
      .trim()
      .split(/\s+/)
      .filter(Boolean)
    argv.push(...extras)
    return argv
  }

  const onSubmit = async () => {
    if (!selectedNode || !isLeaf) {
      message.warning('请选择可执行的叶子命令')
      return
    }
    try {
      const values = await form.validateFields()
      // 手动检查 required positionals（Select/Input）
      for (const opt of selectedNode.options) {
        if (!opt.required) continue
        if (opt.is_flag) continue
        const v = values[opt.dest]
        if (v === undefined || v === null || v === '') {
          message.error(`请填写必填项: ${opt.dest}`)
          return
        }
      }
      const argv = buildArgv(values)
      const firstPos = selectedNode.options.find((o) => o.is_positional)
      const titleParts = [...selectedPath]
      if (firstPos && values[firstPos.dest]) {
        titleParts.push(String(values[firstPos.dest]))
      }
      setSubmitting(true)
      const meta = await submitJob({
        kind: 'cli',
        argv,
        title: titleParts.join(' '),
      })
      setLastJobId(meta.job_id)
      message.success(`已提交任务 ${meta.job_id}`)
    } catch (e) {
      if (e && typeof e === 'object' && 'errorFields' in e) {
        // form validate
        return
      }
      message.error(e instanceof Error ? e.message : String(e))
    } finally {
      setSubmitting(false)
    }
  }

  const renderField = (opt: CliOption) => {
    const label = opt.is_positional
      ? opt.dest
      : pickFlag(opt)
    const help = opt.help ?? undefined

    if (opt.is_flag) {
      return (
        <Form.Item
          key={opt.dest}
          name={opt.dest}
          label={label}
          valuePropName="checked"
          initialValue={Boolean(opt.default)}
          tooltip={help}
        >
          <Switch />
        </Form.Item>
      )
    }

    if (opt.choices && opt.choices.length > 0) {
      return (
        <Form.Item
          key={opt.dest}
          name={opt.dest}
          label={label}
          rules={opt.required ? [{ required: true, message: '必填' }] : []}
          tooltip={help}
          initialValue={
            opt.default != null && opt.choices.includes(String(opt.default))
              ? String(opt.default)
              : undefined
          }
        >
          <Select
            allowClear={!opt.required}
            placeholder={opt.default != null ? String(opt.default) : '请选择'}
            options={opt.choices.map((c) => ({ value: c, label: c }))}
          />
        </Form.Item>
      )
    }

    // positional without choices → Input (or Select if choices handled)
    if (opt.is_positional) {
      return (
        <Form.Item
          key={opt.dest}
          name={opt.dest}
          label={label}
          rules={opt.required ? [{ required: true, message: '必填' }] : []}
          tooltip={help}
        >
          <Input placeholder={opt.default != null ? String(opt.default) : ''} />
        </Form.Item>
      )
    }

    return (
      <Form.Item
        key={opt.dest}
        name={opt.dest}
        label={label}
        rules={opt.required ? [{ required: true, message: '必填' }] : []}
        tooltip={help}
      >
        <Input
          placeholder={
            opt.default != null && opt.default !== 'None'
              ? `默认: ${String(opt.default)}（留空不传）`
              : '留空不传'
          }
        />
      </Form.Item>
    )
  }

  return (
    <div>
      <Typography.Title level={4} style={{ marginTop: 0 }}>
        命令启动器
      </Typography.Title>
      <Typography.Paragraph type="secondary" style={{ marginTop: -8 }}>
        从真实 argparse 树生成表单，提交为后台 cli 任务（不阻塞浏览器）。
      </Typography.Paragraph>

      <Row gutter={16}>
        <Col xs={24} md={10} lg={8}>
          <Card size="small" title="命令树" loading={loading}>
            {schema ? (
              <Tree
                treeData={treeData}
                onSelect={(keys) => {
                  const k = String(keys[0] ?? '')
                  if (!k) return
                  setSelectedPath(k.split('/').filter(Boolean))
                }}
                selectedKeys={
                  selectedPath.length ? [nodeKey(selectedPath)] : []
                }
                defaultExpandAll={false}
                height={560}
                style={{ overflow: 'auto' }}
              />
            ) : (
              <Empty description="加载 schema…" />
            )}
          </Card>
        </Col>
        <Col xs={24} md={14} lg={16}>
          <Card
            size="small"
            title={
              selectedPath.length
                ? `fz ${selectedPath.join(' ')}`
                : '选择命令'
            }
          >
            {!selectedNode || !isLeaf ? (
              <Empty description="请在左侧选择可执行的叶子命令" />
            ) : (
              <>
                {selectedNode.help ? (
                  <Typography.Paragraph type="secondary">
                    {selectedNode.help}
                  </Typography.Paragraph>
                ) : null}
                <Form form={form} layout="vertical" size="middle">
                  {selectedNode.options.map(renderField)}
                  <Form.Item label="附加参数（高级，按空格拆分）">
                    <Input.TextArea
                      value={extra}
                      onChange={(e) => setExtra(e.target.value)}
                      rows={2}
                      placeholder="例如 --set key=value"
                    />
                  </Form.Item>
                  <Space>
                    <Button
                      type="primary"
                      loading={submitting}
                      onClick={() => void onSubmit()}
                    >
                      提交任务
                    </Button>
                    {lastJobId ? (
                      <Typography.Text>
                        已提交{' '}
                        <Typography.Text code>{lastJobId}</Typography.Text>
                        {' · '}
                        <Link to="/jobs">去任务中心</Link>
                      </Typography.Text>
                    ) : null}
                  </Space>
                </Form>
              </>
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}
