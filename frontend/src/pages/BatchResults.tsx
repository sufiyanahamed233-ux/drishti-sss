import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet'
import {
  AlertCircle,
  ArrowLeft,
  Download,
  ExternalLink,
  FileCode,
  FileText,
  Layers,
  MapPin,
  RefreshCw,
  UploadCloud,
} from 'lucide-react'
import {
  getBatch,
  getBatchReportGeoJson,
  getBatchReportJson,
  getBatchReportPdfUrl,
} from '../services/api'
import type { DetectionResult, InvestigationBatchResult, ScanResult } from '../types/api'

// Fix Leaflet marker icons
delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

interface MappedDetection {
  scan: ScanResult
  detection: DetectionResult & { target_latitude: number; target_longitude: number }
  detectionIndex: number
}

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

function BatchSurveyMap({ scans }: { scans: ScanResult[] }) {
  const navigate = useNavigate()

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

  return (
    <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900 overflow-hidden">
      <div className="border-b border-slate-800 p-5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MapPin className="h-5 w-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Batch Survey Detection Map</h2>
        </div>
        <span className="text-xs text-slate-400">
          {mapped.length} georeferenced detection{mapped.length !== 1 ? 's' : ''} across {scans.length} scan{scans.length !== 1 ? 's' : ''}
        </span>
      </div>

      {mapped.length === 0 ? (
        <div className="p-8 text-center text-slate-400 text-sm">
          No georeferenced detections are available to display on the map for this batch.
        </div>
      ) : (
        <MapContainer
          center={positions[0] || [0, 0]}
          zoom={14}
          style={{ height: '450px', width: '100%' }}
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
                <div style={{ minWidth: '220px', fontSize: '13px', lineHeight: 1.6 }}>
                  <strong style={{ fontSize: '14px', color: '#0f172a' }}>{m.scan.scan_identity}</strong>
                  <br />
                  <span style={{ color: '#64748b', fontSize: '11px' }}>
                    Target #{m.detectionIndex + 1}
                  </span>
                  <br />
                  <span style={{ color: '#0891b2', fontWeight: 600 }}>{m.detection.class_name}</span>
                  <br />
                  Confidence: <strong>{(m.detection.confidence * 100).toFixed(1)}%</strong>
                  <br />
                  Lat: <strong>{m.detection.target_latitude.toFixed(6)}</strong>
                  <br />
                  Lon: <strong>{m.detection.target_longitude.toFixed(6)}</strong>
                  <br />
                  Slant Range:{' '}
                  <strong>
                    {m.detection.range_m != null ? `${m.detection.range_m.toFixed(1)} m` : '—'}
                  </strong>
                  <br />
                  Ground Range:{' '}
                  <strong>
                    {m.detection.ground_range_m != null
                      ? `${m.detection.ground_range_m.toFixed(1)} m`
                      : '—'}
                  </strong>
                  <br />
                  <button
                    type="button"
                    onClick={() => navigate(`/scans/${m.scan.id}`)}
                    style={{
                      marginTop: '8px',
                      display: 'inline-block',
                      padding: '4px 10px',
                      backgroundColor: '#0891b2',
                      color: '#ffffff',
                      borderRadius: '4px',
                      fontSize: '12px',
                      fontWeight: 600,
                      border: 'none',
                      cursor: 'pointer',
                    }}
                  >
                    View Scan Detail →
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

export default function BatchResults() {
  const { batchId } = useParams<{ batchId: string }>()
  const navigate = useNavigate()

  const [batch, setBatch] = useState<InvestigationBatchResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [downloadingJson, setDownloadingJson] = useState(false)
  const [downloadingGeoJson, setDownloadingGeoJson] = useState(false)

  const reloadBatch = async (id: string) => {
    setLoading(true)
    setError(null)
    try {
      const data = await getBatch(id)
      setBatch(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to load batch ${id}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    async function loadBatch(id: string) {
      try {
        const data = await getBatch(id)
        setBatch(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : `Failed to load batch ${id}`)
      } finally {
        setLoading(false)
      }
    }

    if (batchId) {
      loadBatch(batchId)
    }
  }, [batchId])

  const mappedDetectionsCount = useMemo(() => {
    if (!batch) return 0
    let count = 0
    batch.scans.forEach((scan) => {
      scan.detections.forEach((det) => {
        if (det.target_latitude !== null && det.target_longitude !== null) {
          count++
        }
      })
    })
    return count
  }, [batch])

  const handleDownloadJson = async () => {
    if (!batchId) return
    setDownloadingJson(true)
    try {
      const data = await getBatchReportJson(batchId)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `drishti_sss_batch_${batchId}_report.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      alert(`Failed to download JSON report: ${err instanceof Error ? err.message : err}`)
    } finally {
      setDownloadingJson(false)
    }
  }

  const handleDownloadGeoJson = async () => {
    if (!batchId) return
    setDownloadingGeoJson(true)
    try {
      const data = await getBatchReportGeoJson(batchId)
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/geo+json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `drishti_sss_batch_${batchId}_report.geojson`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      alert(`Failed to download GeoJSON report: ${err instanceof Error ? err.message : err}`)
    } finally {
      setDownloadingGeoJson(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 p-8 text-slate-300 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <RefreshCw className="h-5 w-5 animate-spin text-cyan-400" />
          <span>Loading investigation batch {batchId}...</span>
        </div>
      </div>
    )
  }

  if (error || !batch) {
    return (
      <main className="min-h-screen bg-slate-950 p-8 text-white">
        <div className="mx-auto max-w-4xl">
          <Link
            to="/"
            className="mb-6 inline-flex items-center gap-2 text-sm font-medium text-slate-400 hover:text-cyan-400"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>

          <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-6 text-red-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-6 w-6 text-red-400 shrink-0" />
              <div>
                <h2 className="text-lg font-bold text-red-300">Investigation Batch Not Found</h2>
                <p className="mt-1 text-sm">{error || `Batch '${batchId}' does not exist.`}</p>
                <div className="mt-4 flex gap-3">
                  <button
                    type="button"
                    onClick={() => batchId && reloadBatch(batchId)}
                    className="rounded-lg bg-red-600 px-4 py-2 text-xs font-semibold text-white hover:bg-red-500 cursor-pointer"
                  >
                    Retry
                  </button>
                  <Link
                    to="/batch-upload"
                    className="rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-200 hover:bg-slate-700 hover:text-white"
                  >
                    Upload New Batch
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-slate-950 p-6 md:p-8 text-white">
      <div className="mx-auto max-w-7xl">
        {/* Navigation Breadcrumb */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm font-medium text-slate-400 transition hover:text-cyan-400"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>

          <div className="flex items-center gap-3">
            <Link
              to="/batch-upload"
              className="inline-flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900 px-3.5 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-800 hover:text-white"
            >
              <UploadCloud className="h-3.5 w-3.5 text-cyan-400" />
              Upload Another Batch
            </Link>
          </div>
        </div>

        {/* Page Header */}
        <header className="mb-8 rounded-xl border border-slate-800 bg-slate-900 p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="mb-1 text-xs font-semibold uppercase tracking-[0.25em] text-cyan-400">
                DRISHTI • INVESTIGATION BATCH
              </p>
              <h1 className="text-2xl md:text-3xl font-bold font-mono text-white">
                {batch.batch_id}
              </h1>
              <p className="mt-2 text-sm text-slate-400">
                Processed on {new Date(batch.created_at).toLocaleString()} • Provenance Source:{' '}
                <span className="font-semibold text-cyan-300 uppercase">{batch.data_source}</span>
              </p>
            </div>

            {/* Report Actions */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleDownloadJson}
                disabled={downloadingJson}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white cursor-pointer"
                title="Download complete structured JSON investigation report"
              >
                <FileCode className="h-3.5 w-3.5 text-cyan-400" />
                {downloadingJson ? 'Downloading...' : 'JSON Report'}
              </button>

              <button
                type="button"
                onClick={handleDownloadGeoJson}
                disabled={downloadingGeoJson}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white cursor-pointer"
                title="Download RFC 7946 GeoJSON FeatureCollection of all detections"
              >
                <MapPin className="h-3.5 w-3.5 text-emerald-400" />
                {downloadingGeoJson ? 'Downloading...' : 'GeoJSON Report'}
              </button>

              <a
                href={getBatchReportPdfUrl(batch.batch_id)}
                download={`drishti_sss_batch_${batch.batch_id}_report.pdf`}
                className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-600 px-3.5 py-2 text-xs font-semibold text-white shadow-lg shadow-cyan-950/50 transition hover:bg-cyan-500 cursor-pointer"
                title="Download official PDF report with provenance and georeference tables"
              >
                <Download className="h-3.5 w-3.5" />
                PDF Report
              </a>
            </div>
          </div>
        </header>

        {/* Summary Cards */}
        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Scans</p>
            <p className="mt-2 text-3xl font-bold text-white">{batch.total_scans}</p>
            <p className="mt-1 text-xs text-slate-500">Ingested from batch</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Successful Scans</p>
            <p className="mt-2 text-3xl font-bold text-emerald-400">{batch.successful_scans}</p>
            <p className="mt-1 text-xs text-slate-500">100% processed without failure</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Detections</p>
            <p className="mt-2 text-3xl font-bold text-cyan-400">{batch.total_detections}</p>
            <p className="mt-1 text-xs text-slate-500">Targets identified by YOLO</p>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Mapped Detections</p>
            <p className="mt-2 text-3xl font-bold text-indigo-400">{mappedDetectionsCount}</p>
            <p className="mt-1 text-xs text-slate-500">Deterministic georeferenced coordinates</p>
          </div>
        </section>

        {/* Detection Class Summary */}
        <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900 p-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-cyan-400" />
              <h2 className="text-base font-semibold text-white">Detection Summary by Target Class</h2>
            </div>
            <span className="text-xs text-slate-400">
              {Object.keys(batch.class_counts).length} Class{Object.keys(batch.class_counts).length !== 1 ? 'es' : ''} Present
            </span>
          </div>

          {Object.keys(batch.class_counts).length === 0 ? (
            <p className="mt-4 text-sm text-slate-400">No target detections found across scans in this batch.</p>
          ) : (
            <div className="mt-4 grid gap-3 sm:grid-cols-2 md:grid-cols-4">
              {Object.entries(batch.class_counts).map(([className, count]) => (
                <div
                  key={className}
                  className="flex items-center justify-between rounded-lg bg-slate-950/60 p-3.5 border border-slate-800/80"
                >
                  <span className="font-medium text-sm text-slate-200 capitalize">
                    {className.replace(/_/g, ' ')}
                  </span>
                  <span className="rounded-md bg-cyan-950/80 px-2 py-0.5 text-xs font-bold text-cyan-300 border border-cyan-800/50">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Survey Map */}
        <BatchSurveyMap scans={batch.scans} />

        {/* Scan Table */}
        <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900 overflow-hidden">
          <div className="border-b border-slate-800 p-5 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="h-5 w-5 text-cyan-400" />
              <h2 className="text-base font-semibold text-white">Batch Scans List</h2>
            </div>
            <span className="text-xs text-slate-400">
              {batch.scans.length} Scan{batch.scans.length !== 1 ? 's' : ''} (including zero-detection scans)
            </span>
          </div>

          {batch.scans.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-sm">
              No scans found in this batch.
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {batch.scans.map((scan) => (
                <div
                  key={scan.id}
                  className="flex flex-col gap-4 p-5 transition-colors hover:bg-slate-800/40 md:flex-row md:items-center md:justify-between"
                >
                  <div>
                    <div className="flex items-center gap-2.5">
                      <span className="font-semibold text-white text-base">
                        {scan.scan_identity}
                      </span>
                      {scan.detection_count === 0 ? (
                        <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-[11px] font-medium text-slate-400">
                          0 detections
                        </span>
                      ) : (
                        <span className="rounded-full bg-cyan-950 px-2.5 py-0.5 text-[11px] font-bold text-cyan-300 border border-cyan-800/40">
                          {scan.detection_count} detection{scan.detection_count !== 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                    <p className="mt-1 text-xs text-slate-400">
                      Acquisition: {new Date(scan.timestamp).toLocaleString()} • Heading: {scan.heading.toFixed(1)}° • Altitude: {scan.altitude.toFixed(1)}m
                    </p>
                  </div>

                  <div className="flex flex-wrap items-center gap-6 text-sm">
                    <div>
                      <span className="text-xs text-slate-500 block">Source</span>
                      <span className="font-semibold uppercase text-xs text-slate-300">
                        {scan.data_source}
                      </span>
                    </div>

                    <div>
                      <span className="text-xs text-slate-500 block">Sonar Position</span>
                      <span className="font-mono text-xs font-semibold text-slate-300">
                        {scan.sonar_latitude.toFixed(5)}, {scan.sonar_longitude.toFixed(5)}
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={() => navigate(`/scans/${scan.id}`)}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white cursor-pointer"
                    >
                      Scan Detail
                      <ExternalLink className="h-3 w-3" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Footer Navigation */}
        <div className="mt-8 flex justify-between items-center text-xs text-slate-500">
          <span>DRISHTI-SSS • Autonomous Sonar Investigation System</span>
          <Link to="/" className="text-cyan-400 hover:underline">
            Return to Dashboard
          </Link>
        </div>
      </div>
    </main>
  )
}
