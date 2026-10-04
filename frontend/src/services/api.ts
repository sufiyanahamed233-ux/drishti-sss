import type { ScanResult } from '../types/api'

const API_BASE_URL = 'http://127.0.0.1:8000/api'

export async function getScans(): Promise<ScanResult[]> {
  const response = await fetch(`${API_BASE_URL}/scans`)

  if (!response.ok) {
    throw new Error(`Failed to fetch scans: ${response.status}`)
  }

  return response.json()
}

export async function getScan(scanId: number): Promise<ScanResult> {
  const response = await fetch(`${API_BASE_URL}/scans/${scanId}`)

  if (!response.ok) {
    throw new Error(`Failed to fetch scan ${scanId}: ${response.status}`)
  }

  return response.json()
}
