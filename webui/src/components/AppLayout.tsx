import { useEffect, useMemo, useState } from 'react'
import { Layout, Menu, Spin, Tooltip, Typography } from 'antd'
import type { MenuProps } from 'antd'
import {
  AppstoreOutlined,
  BankOutlined,
  CodeOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  FolderOpenOutlined,
  FundOutlined,
  LineChartOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { fetchHealth } from '../api/client'
import {
  DOMAIN_GROUPS,
  domainDesc,
  domainGroup,
  domainLabel,
} from '../domainMeta'

const { Header, Sider, Content } = Layout

type MenuItem = Required<MenuProps>['items'][number]

export function AppLayout() {
  const location = useLocation()
  const [domains, setDomains] = useState<string[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then((h) => {
        if (!cancelled) setDomains(h.domains)
      })
      .catch(() => {
        if (!cancelled) setDomains([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const selectedKey = (() => {
    const path = location.pathname
    if (path.startsWith('/domain/')) {
      return path.split('/')[2] ?? 'overview'
    }
    if (path.startsWith('/run/')) {
      return path.split('/')[2] ?? 'overview'
    }
    if (path.startsWith('/library')) return 'library'
    if (path.startsWith('/store')) return 'store'
    if (path.startsWith('/ops')) return 'ops'
    if (path.startsWith('/reports')) return 'reports'
    if (path.startsWith('/files')) return 'files'
    if (path.startsWith('/jobs')) return 'jobs'
    if (path.startsWith('/cli')) return 'cli'
    if (path.startsWith('/strategy')) return 'strategy'
    return 'overview'
  })()

  const topKeys = new Set([
    'overview',
    'library',
    'store',
    'ops',
    'reports',
    'files',
    'jobs',
    'cli',
    'strategy',
  ])

  const openKeys = (() => {
    if (!topKeys.has(selectedKey)) {
      return ['runs']
    }
    return undefined
  })()

  const domainChildren: MenuItem[] = useMemo(() => {
    const byGroup = new Map<string, string[]>()
    for (const g of DOMAIN_GROUPS) {
      byGroup.set(g, [])
    }
    byGroup.set('其他', [])

    for (const d of domains) {
      const g = domainGroup(d)
      if (!byGroup.has(g)) byGroup.set(g, [])
      byGroup.get(g)!.push(d)
    }

    const items: MenuItem[] = []
    const groupOrder = [...DOMAIN_GROUPS, '其他'] as const
    for (const g of groupOrder) {
      const keys = byGroup.get(g) ?? []
      if (keys.length === 0) continue
      items.push({
        type: 'group',
        key: `group-${g}`,
        label: g,
        children: keys.map((d) => {
          const label = domainLabel(d)
          const desc = domainDesc(d)
          return {
            key: d,
            title: desc || d,
            label: (
              <Tooltip title={desc || d} placement="right">
                <Link to={`/domain/${d}`}>{label}</Link>
              </Tooltip>
            ),
          }
        }),
      })
    }
    return items
  }, [domains])

  const menuItems: MenuItem[] = [
    {
      key: 'overview',
      icon: <AppstoreOutlined />,
      label: <Link to="/">总览</Link>,
    },
    {
      key: 'jobs',
      icon: <ThunderboltOutlined />,
      label: <Link to="/jobs">任务中心</Link>,
    },
    {
      key: 'cli',
      icon: <CodeOutlined />,
      label: <Link to="/cli">命令启动器</Link>,
    },
    {
      key: 'strategy',
      icon: <LineChartOutlined />,
      label: <Link to="/strategy">策略回测</Link>,
    },
    {
      key: 'files',
      icon: <FolderOpenOutlined />,
      label: <Link to="/files">文件管理</Link>,
    },
    {
      key: 'runs',
      icon: <DatabaseOutlined />,
      label: '产物 Runs',
      children: domainChildren,
    },
    {
      key: 'library',
      icon: <FundOutlined />,
      label: <Link to="/library">因子库</Link>,
    },
    {
      key: 'store',
      icon: <BankOutlined />,
      label: <Link to="/store">因子资产</Link>,
    },
    {
      key: 'ops',
      icon: <ExperimentOutlined />,
      label: <Link to="/ops">运营</Link>,
    },
    {
      key: 'reports',
      icon: <FileTextOutlined />,
      label: <Link to="/reports">报告</Link>,
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        theme="light"
        width={240}
        style={{ borderRight: '1px solid #f0f0f0' }}
      >
        <div
          style={{
            padding: '16px 20px',
            fontWeight: 600,
            fontSize: 16,
            borderBottom: '1px solid #f0f0f0',
          }}
        >
          FactorZen
        </div>
        {loading ? (
          <div style={{ padding: 24, textAlign: 'center' }}>
            <Spin size="small" />
          </div>
        ) : (
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            defaultOpenKeys={openKeys ?? ['runs']}
            items={menuItems}
            style={{ borderInlineEnd: 'none' }}
          />
        )}
      </Sider>
      <Layout>
        <Header
          style={{
            background: '#fff',
            padding: '0 24px',
            borderBottom: '1px solid #f0f0f0',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          <Typography.Title level={4} style={{ margin: 0 }}>
            FactorZen
          </Typography.Title>
          <Typography.Text type="secondary" style={{ marginLeft: 12 }}>
            研究产物浏览器
          </Typography.Text>
        </Header>
        <Content style={{ padding: 24, background: '#f5f5f5' }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  )
}
