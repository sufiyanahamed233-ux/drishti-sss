import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet'
import { getScans } from '../services/api'
import type { DetectionResult, ScanResult } from '../types/api'

// Fix Leaflet's default marker icon path broken by bundlers
delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

// ── Survey Detection Map ──────────────────────────────────────────────────────

interface MappedDetection {
  scan: ScanResult
  detection: DetectionResult & { target_latitude: number; target_longitude: number }
  detectionIndex: number
}

/** Fits the map to all marker positions. Child of <MapContainer>. */
function FitBounds({ positions }: { positions: [number, number][] }) {
  const map = useMap()
  useEffect(() => {
    if (positions.length === 0) return
    if (positions.length === 1) {
      map.setView(positions[0], 14)
    } else {
      map.fitBounds(L.latLngBounds(positions), { padding: [40, 40] })
    }
  }, [map, positions])
  return null
}

function SurveyMap({ scans }: { scans: ScanResult[] }) {
  const navigate = useNavigate()

  // Collect every detection with valid georeferenced coordinates across all scans
  const mapped: MappedDetection[] = useMemo(() => {
    const results: MappedDetection[] = []
    scans.forEach((scan) => {
      scan.detections.forEach((det, idx) => {
        if (det.target_latitude !== null && det.target_longitude !== null) {
          results.push({
            scan,
            detection: det as DetectionResult & {
              target_latitude: number
              target_longitude: number
            },
            detectionIndex: idx,
          })
        }
      })
    })
    return results
  }, [scans])

  const positions: [number, number][] = mapped.map((m) => [
    m.detection.target_latitude,
    m.detection.target_longitude,
  ])

  const uniqueScanCount = useMemo(
    () => new Set(mapped.map((m) => m.scan.id)).size,
    [mapped],
  )

  return (
    <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900 overflow-hidden">
      <div className="border-b border-slate-800 p-5 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Survey Detection Map</h2>
        <span className="text-xs text-slate-500">
          {mapped.length} detection{mapped.length !== 1 ? 's' : ''} mapped from{' '}
          {uniqueScanCount} scan{uniqueScanCount !== 1 ? 's' : ''}
        </span>
      </div>

      {mapped.length === 0 ? (
        <div className="p-8 text-center text-slate-400 text-sm">
          {scans.length === 0
            ? 'No scans processed yet — no positions to display.'
            : 'No detections have georeferenced coordinates available across all scans.'}
        </div>
      ) : (
        <MapContainer
          center={[0, 0]}
          zoom={14}
          style={{ height: '480px', width: '100%' }}
          scrollWheelZoom
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds positions={positions} />
          {mapped.map((m) => (
            <Marker
              key={`${m.scan.id}-${m.detection.id ?? m.detectionIndex}`}
              position={[m.detection.target_latitude, m.detection.target_longitude]}
            >
              <Popup>
                <div style={{ minWidth: '200px', fontSize: '13px', lineHeight: 1.7 }}>
                  <strong style={{ fontSize: '14px' }}>{m.scan.scan_identity}</strong>
                  <br />
                  <span style={{ color: '#64748b', fontSize: '11px' }}>
                    Detection {m.detectionIndex + 1}
                  </span>
                  <br />
                  <span style={{ color: '#0e7490' }}>{m.detection.class_name}</span>
                  <br />
                  Confidence:{' '}
                  <strong>{(m.detection.confidence * 100).toFixed(1)}%</strong>
                  <br />
                  Lat: <strong>{m.detection.target_latitude.toFixed(6)}</strong>
                  <br />
                  Lon: <strong>{m.detection.target_longitude.toFixed(6)}</strong>
                  <br />
                  Range:{' '}
                  <strong>
                    {m.detection.range_m != null
                      ? `${m.detection.range_m.toFixed(1)} m`
                      : '—'}
                  </strong>
                  <br />
                  Ground range:{' '}
                  <strong>
                    {m.detection.ground_range_m != null
                      ? `${m.detection.ground_range_m.toFixed(1)} m`
                      : '—'}
                  </strong>
                  <br />
                  <button
                    onClick={() => navigate(`/scans/${m.scan.id}`)}
                    style={{
                      marginTop: '8px',
                      display: 'inline-block',
                      padding: '3px 10px',
                      backgroundColor: '#0e7490',
                      color: '#fff',
                      borderRadius: '4px',
                      fontSize: '12px',
                      fontWeight: 600,
                      border: 'none',
                      cursor: 'pointer',
                    }}
                  >
                    View scan →
                  </button>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      )}
    </section>
  )
}

// ── Dashboard Page ────────────────────────────────────────────────────────────

function Dashboard() {
  const [scans, setScans] = useState<ScanResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadScans() {
      try {
        const data = await getScans()
        setScans(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load scans')
      } finally {
        setLoading(false)
      }
    }

    loadScans()
  }, [])

  const totalDetections = useMemo(
    () => scans.reduce((total, scan) => total + scan.detection_count, 0),
    [scans],
  )

  const classCounts = useMemo(() => {
    const counts: Record<string, number> = {}

    scans.forEach((scan) => {
      scan.detections.forEach((detection) => {
        counts[detection.class_name] = (counts[detection.class_name] ?? 0) + 1
      })
    })

    return counts
  }, [scans])

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 p-8 text-slate-300">
        Loading sonar investigations...
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-950 p-8">
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error}
        </div>
      </div>
    )
  }

  return (
    <main className="min-h-screen bg-slate-950 p-8 text-white">
      <div className="mx-auto max-w-7xl">
        <header className="mb-8">
          <p className="mb-2 text-sm font-medium uppercase tracking-[0.25em] text-cyan-400">
            DRISHTI • SONAR INTELLIGENCE
          </p>

          <h1 className="text-3xl font-bold">
            Marine Debris &amp; Anomaly Investigation
          </h1>

          <p className="mt-2 text-slate-400">
            AI-assisted side-scan sonar detection and deterministic georeferencing.
          </p>
        </header>

        {/* ── Stats ── */}
        <section className="grid gap-4 md:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">Total Scans</p>
            <p className="mt-2 text-3xl font-bold">{scans.length}</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">Total Detections</p>
            <p className="mt-2 text-3xl font-bold">{totalDetections}</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-sm text-slate-400">Detection Classes</p>
            <p className="mt-2 text-3xl font-bold">
              {Object.keys(classCounts).length}
            </p>
          </div>
        </section>

        {/* ── Survey Detection Map ── */}
        <SurveyMap scans={scans} />

        {/* ── Processed Sonar Scans list ── */}
        <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900">
          <div className="border-b border-slate-800 p-5">
            <h2 className="text-lg font-semibold">Processed Sonar Scans</h2>
          </div>

          {scans.length === 0 ? (
            <div className="p-8 text-center text-slate-400">
              No sonar scans have been processed yet.
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {scans.map((scan) => (
                <Link
                  key={scan.id}
                  to={`/scans/${scan.id}`}
                  className="flex cursor-pointer flex-col gap-4 p-5 transition-colors hover:bg-slate-800/60 md:flex-row md:items-center md:justify-between"
                >
                  <div>
                    <p className="font-medium">{scan.scan_identity}</p>
                    <p className="mt-1 text-sm text-slate-400">
                      {new Date(scan.timestamp).toLocaleString()}
                    </p>
                  </div>

                  <div className="flex items-center gap-6 text-sm">
                    <div>
                      <span className="text-slate-500">Detections</span>
                      <p className="font-semibold">{scan.detection_count}</p>
                    </div>

                    <div>
                      <span className="text-slate-500">Source</span>
                      <p className="font-semibold">{scan.data_source}</p>
                    </div>

                    <div>
                      <span className="text-slate-500">Position</span>
                      <p className="font-semibold">
                        {scan.sonar_latitude.toFixed(5)},{' '}
                        {scan.sonar_longitude.toFixed(5)}
                      </p>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  )
}

export default Dashboard
