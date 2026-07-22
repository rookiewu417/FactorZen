import { useEffect, useState } from 'react'
import { Layout, Menu, Spin, Typography } from 'antd'
import {
  AppstoreOutlined,
  DatabaseOutlined,
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
    return 'overview'
  })()

  const menuItems = [
    {
      key: 'overview',
      icon: <AppstoreOutlined />,
      label: <Link to="/">总览</Link>,
    },
    ...domains.map((d) => ({
      key: d,
      icon: <DatabaseOutlined />,
      label: <Link to={`/domain/${d}`}>{d}</Link>,
    })),
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
