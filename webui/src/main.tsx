import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import 'antd/dist/reset.css'
import './index.css'

const root = document.getElementById('root')
if (!root) {
  throw new Error('root 元素不存在')
}

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
