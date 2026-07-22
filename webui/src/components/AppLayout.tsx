import { useEffect, useState } from 'react'
import { Layout, Menu, Spin, Typography } from 'antd'
import {
  AppstoreOutlined,
  BankOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  FundOutlined,
} from '@ant-design/icons'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { fetchHealth } from '../api/client'

const { Header, Sider, Content } = Layout

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
    return 'overview'
  })()

  const openKeys = (() => {
    if (
      selectedKey !== 'overview' &&
      selectedKey !== 'library' &&
      selectedKey !== 'store' &&
      selectedKey !== 'ops' &&
      selectedKey !== 'reports'
    ) {
      return ['runs']
    }
    return undefined
  })()

  const menuItems = [
    {
      key: 'overview',
      icon: <AppstoreOutlined />,
      label: <Link to="/">总览</Link>,
    },
    {
      key: 'runs',
      icon: <DatabaseOutlined />,
      label: '产物 Runs',
      children: domains.map((d) => ({
        key: d,
        label: <Link to={`/domain/${d}`}>{d}</Link>,
      })),
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
        width={220}
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
