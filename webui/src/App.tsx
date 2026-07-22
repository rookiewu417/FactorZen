import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { DomainListPage } from './pages/DomainList'
import { OverviewPage } from './pages/Overview'
import { RunDetailPage } from './pages/RunDetail'

export default function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <HashRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<OverviewPage />} />
            <Route path="domain/:domain" element={<DomainListPage />} />
            <Route path="run/:domain/:runId" element={<RunDetailPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </ConfigProvider>
  )
}
