import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import ScanDetail from './pages/ScanDetail'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/scans/:scanId" element={<ScanDetail />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
