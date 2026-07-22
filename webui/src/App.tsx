import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './components/AppLayout'
import { DomainListPage } from './pages/DomainList'
import { FilesPage } from './pages/Files'
import { LibraryPage } from './pages/Library'
import { OpsPage } from './pages/Ops'
import { OverviewPage } from './pages/Overview'
import { ReportsPage } from './pages/Reports'
import { RunDetailPage } from './pages/RunDetail'
import { StorePage } from './pages/Store'

export default function App() {
  return (
    <ConfigProvider locale={zhCN}>
      <HashRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<OverviewPage />} />
            <Route path="domain/:domain" element={<DomainListPage />} />
            <Route path="run/:domain/:runId" element={<RunDetailPage />} />
            <Route path="library" element={<LibraryPage />} />
            <Route path="store" element={<StorePage />} />
            <Route path="ops" element={<OpsPage />} />
            <Route path="reports" element={<ReportsPage />} />
            <Route path="files" element={<FilesPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </ConfigProvider>
  )
}
