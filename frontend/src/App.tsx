import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import ScanDetail from './pages/ScanDetail'
import BatchUpload from './pages/BatchUpload'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/batch-upload" element={<BatchUpload />} />
        <Route path="/scans/:scanId" element={<ScanDetail />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
