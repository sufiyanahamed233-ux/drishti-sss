import { useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'
import { MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet'
import { getScan } from '../services/api'
import type { DetectionResult, ScanResult } from '../types/api'

// Fix Leaflet's default marker icon path broken by bundlers
delete (L.Icon.Default.prototype as unknown as Record<string, unknown>)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
})

const API_BASE_URL = 'http://127.0.0.1:8000/api'

// ── Colour palette cycling for bbox borders/labels ──────────────────────────
const BOX_COLOURS = [
  'rgba(6,182,212,0.9)',   // cyan-500
  'rgba(168,85,247,0.9)',  // purple-500
  'rgba(234,179,8,0.9)',   // yellow-500
  'rgba(34,197,94,0.9)',   // green-500
  'rgba(249,115,22,0.9)',  // orange-500
  'rgba(239,68,68,0.9)',   // red-500
]

function boxColour(index: number): string {
  return BOX_COLOURS[index % BOX_COLOURS.length]
}

// ── Sonar Image Viewer ───────────────────────────────────────────────────────

interface ImageViewerProps {
  scanId: number
  detections: DetectionResult[]
}

function SonarImageViewer({ scanId, detections }: ImageViewerProps) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null)
  const [imgError, setImgError] = useState(false)
  const [imgLoading, setImgLoading] = useState(true)

  const imageUrl = `${API_BASE_URL}/scans/${scanId}/image`

  function handleLoad() {
    const img = imgRef.current
    if (img) {
      setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
    }
    setImgLoading(false)
  }

  function handleError() {
    setImgError(true)
    setImgLoading(false)
  }

  // Convert absolute pixel bbox → percentage positions relative to the
  // natural image dimensions so the overlays scale with the displayed image.
  function toPercent(bbox: DetectionResult['bbox']) {
    if (!naturalSize) return null
    const { w, h } = naturalSize
    return {
      left:   `${(bbox.x1 / w) * 100}%`,
      top:    `${(bbox.y1 / h) * 100}%`,
      width:  `${((bbox.x2 - bbox.x1) / w) * 100}%`,
      height: `${((bbox.y2 - bbox.y1) / h) * 100}%`,
    }
  }

  return (
    <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900">
      {/* Header — OUTSIDE the coordinate wrapper */}
      <div className="border-b border-slate-800 p-5 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Sonar Image</h2>
        {naturalSize && (
          <span className="text-xs text-slate-500 font-mono">
            {naturalSize.w} × {naturalSize.h} px
          </span>
        )}
      </div>

      <div className="p-4">
        {/* Error state — OUTSIDE the coordinate wrapper */}
        {imgError && (
          <div className="flex items-center justify-center rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-8 text-sm text-yellow-300">
            Image file not available on disk for this scan.
          </div>
        )}

        {/* Loading placeholder — OUTSIDE the coordinate wrapper */}
        {!imgError && imgLoading && (
          <div className="flex items-center justify-center rounded-lg bg-slate-800 p-12 text-sm text-slate-400">
            Loading sonar image…
          </div>
        )}

        {/*
         * COORDINATE WRAPPER — must contain ONLY the <img> and its overlay.
         *
         * Rules that make the math exact:
         *   position:relative  establishes the containing block for the overlay
         *   width:100%         fills the available column
         *   height is NOT set  → the browser derives it from the <img> child
         *
         * The <img> has display:block + width:100% + height:auto (no object-fit)
         * so the element's own bounding box IS the rendered image pixels.
         * Therefore overlay position:absolute left:0 top:0 width:100% height:100%
         * covers exactly those pixels at every viewport width.
         *
         * display:none while loading keeps the wrapper out of the layout
         * entirely, preventing the placeholder div from adding any height.
         */}
        {!imgError && (
          <div
            style={{
              position: 'relative',
              width: '100%',
              display: imgLoading ? 'none' : 'block',
            }}
          >
            <img
              ref={imgRef}
              src={imageUrl}
              alt="Sonar scan image"
              onLoad={handleLoad}
              onError={handleError}
              style={{
                display: 'block',
                width: '100%',
                height: 'auto',
                borderRadius: '8px',
              }}
            />

            {naturalSize && (
              <div
                style={{
                  position: 'absolute',
                  left: 0,
                  top: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                }}
              >
                {detections.map((det, idx) => {
                  const pos = toPercent(det.bbox)
                  if (!pos) return null
                  const colour = boxColour(idx)
                  return (
                    <div
                      key={det.id ?? idx}
                      style={{
                        position: 'absolute',
                        left: pos.left,
                        top: pos.top,
                        width: pos.width,
                        height: pos.height,
                        border: `2px solid ${colour}`,
                        boxSizing: 'border-box',
                      }}
                    >
                      {/* Label pinned inside the top-left corner of the box */}
                      <span
                        style={{
                          position: 'absolute',
                          left: 0,
                          top: 0,
                          backgroundColor: colour,
                          color: '#0f172a',
                          fontSize: '10px',
                          fontWeight: 700,
                          lineHeight: 1.2,
                          padding: '1px 4px',
                          borderRadius: '2px',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {det.class_name} {(det.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* Legend */}
        {!imgError && !imgLoading && detections.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-3">
            {detections.map((det, idx) => (
              <div key={det.id ?? idx} className="flex items-center gap-1.5 text-xs">
                <span
                  className="inline-block h-3 w-3 rounded-sm flex-shrink-0"
                  style={{ backgroundColor: boxColour(idx) }}
                />
                <span className="text-slate-300">
                  #{idx + 1} {det.class_name} ({(det.confidence * 100).toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        )}

        {!imgError && !imgLoading && detections.length === 0 && (
          <p className="mt-3 text-center text-xs text-slate-500">
            No detections — image shown without overlays.
          </p>
        )}
      </div>
    </section>
  )
}

// ── Georeferenced Detection Map ─────────────────────────────────────────────

interface DetectionMapProps {
  detections: DetectionResult[]
}

/**
 * Fits the map to the bounding box of all plotted markers.
 * Must be a child of <MapContainer> to access the map instance via useMap().
 */
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

function DetectionMap({ detections }: DetectionMapProps) {
  // Only plot detections that have valid georeferenced coordinates
  const mapped = detections.filter(
    (d): d is DetectionResult & { target_latitude: number; target_longitude: number } =>
      d.target_latitude !== null && d.target_longitude !== null,
  )

  const positions: [number, number][] = mapped.map((d) => [d.target_latitude, d.target_longitude])

  // Fallback centre — used only when there are no mapped detections
  const defaultCenter: [number, number] = [0, 0]

  return (
    <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900 overflow-hidden">
      <div className="border-b border-slate-800 p-5 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Georeferenced Detection Map</h2>
        <span className="text-xs text-slate-500">
          {mapped.length} / {detections.length} detection{detections.length !== 1 ? 's' : ''} mapped
        </span>
      </div>

      {mapped.length === 0 ? (
        <div className="p-8 text-center text-slate-400 text-sm">
          {detections.length === 0
            ? 'No detections in this scan — no positions to display.'
            : 'No detections have georeferenced coordinates available.'}
        </div>
      ) : (
        <MapContainer
          center={defaultCenter}
          zoom={14}
          style={{ height: '420px', width: '100%' }}
          scrollWheelZoom
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds positions={positions} />
          {mapped.map((det, idx) => (
            <Marker key={det.id ?? idx} position={[det.target_latitude, det.target_longitude]}>
              <Popup>
                <div style={{ minWidth: '180px', fontSize: '13px', lineHeight: 1.6 }}>
                  <strong>Detection {idx + 1}</strong>
                  <br />
                  <span style={{ color: '#0e7490' }}>{det.class_name}</span>
                  <br />
                  Confidence: <strong>{(det.confidence * 100).toFixed(1)}%</strong>
                  <br />
                  Lat: <strong>{det.target_latitude.toFixed(6)}</strong>
                  <br />
                  Lon: <strong>{det.target_longitude.toFixed(6)}</strong>
                  <br />
                  Range: <strong>{det.range_m != null ? `${det.range_m.toFixed(1)} m` : '—'}</strong>
                  <br />
                  Ground range: <strong>{det.ground_range_m != null ? `${det.ground_range_m.toFixed(1)} m` : '—'}</strong>
                </div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>
      )}
    </section>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

function ScanDetail() {
  const { scanId } = useParams<{ scanId: string }>()
  const [scan, setScan] = useState<ScanResult | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function loadScan() {
      if (!scanId) {
        setError('Scan ID is missing')
        setLoading(false)
        return
      }

      try {
        const data = await getScan(Number(scanId))
        setScan(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load scan')
      } finally {
        setLoading(false)
      }
    }

    loadScan()
  }, [scanId])

  if (loading) {
    return (
      <main className="min-h-screen bg-slate-950 p-8 text-slate-300">
        Loading scan investigation…
      </main>
    )
  }

  if (error || !scan) {
    return (
      <main className="min-h-screen bg-slate-950 p-8">
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error ?? 'Scan not found'}
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-slate-950 p-8 text-white">
      <div className="mx-auto max-w-7xl">
        {/* Breadcrumb */}
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-xs text-slate-500 hover:text-cyan-400 transition-colors"
        >
          ← Dashboard
        </Link>

        <p className="mt-4 text-sm font-medium uppercase tracking-[0.25em] text-cyan-400">
          DRISHTI • SCAN INVESTIGATION
        </p>

        <h1 className="mt-2 text-3xl font-bold">{scan.scan_identity}</h1>

        <p className="mt-2 text-slate-400">
          {new Date(scan.timestamp).toLocaleString()}
        </p>

        {/* ── Sonar image viewer ── */}
        <SonarImageViewer scanId={scan.id} detections={scan.detections} />

        {/* ── Georeferenced detection map ── */}
        <DetectionMap detections={scan.detections} />

        {/* ── Metadata cards ── */}
        <section className="mt-8 grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Navigation Metadata</h2>

            <div className="mt-4 space-y-3 text-sm">
              <p>
                <span className="text-slate-500">Sonar Latitude:</span>{' '}
                {scan.sonar_latitude}
              </p>

              <p>
                <span className="text-slate-500">Sonar Longitude:</span>{' '}
                {scan.sonar_longitude}
              </p>

              <p>
                <span className="text-slate-500">Heading:</span>{' '}
                {scan.heading}°
              </p>

              <p>
                <span className="text-slate-500">Altitude:</span>{' '}
                {scan.altitude} m
              </p>

              <p>
                <span className="text-slate-500">Range:</span>{' '}
                {scan.range ?? 'Per-detection'}
              </p>

              <p>
                <span className="text-slate-500">Range Type:</span>{' '}
                {scan.range_type}
              </p>

              <p>
                <span className="text-slate-500">Source:</span>{' '}
                {scan.data_source}
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
            <h2 className="text-lg font-semibold">Detection Summary</h2>

            <p className="mt-4 text-4xl font-bold">{scan.detection_count}</p>

            <p className="mt-1 text-sm text-slate-400">detected anomalies</p>
          </div>
        </section>

        {/* ── Detection list ── */}
        <section className="mt-8 rounded-xl border border-slate-800 bg-slate-900">
          <div className="border-b border-slate-800 p-5">
            <h2 className="text-lg font-semibold">Detections</h2>
          </div>

          {scan.detections.length === 0 ? (
            <div className="p-8 text-center text-slate-400">
              No anomalies detected in this sonar scan.
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {scan.detections.map((detection, index) => (
                <div key={detection.id ?? index} className="p-5">
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <div className="flex items-center gap-3">
                      {/* Colour swatch matching the image overlay */}
                      <span
                        className="mt-0.5 inline-block h-3 w-3 flex-shrink-0 rounded-sm"
                        style={{ backgroundColor: boxColour(index) }}
                      />
                      <div>
                        <p className="font-semibold">Detection {index + 1}</p>
                        <p className="mt-1 text-cyan-400">{detection.class_name}</p>
                      </div>
                    </div>

                    <div className="text-sm text-slate-400">
                      Confidence:{' '}
                      <span className="font-semibold text-white">
                        {(detection.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="mt-5 grid gap-4 text-sm md:grid-cols-3">
                    <div>
                      <span className="text-slate-500">Relative Bearing</span>
                      <p className="mt-1 font-semibold">
                        {detection.relative_bearing ?? '—'}°
                      </p>
                    </div>

                    <div>
                      <span className="text-slate-500">Absolute Bearing</span>
                      <p className="mt-1 font-semibold">
                        {detection.absolute_bearing ?? '—'}°
                      </p>
                    </div>

                    <div>
                      <span className="text-slate-500">Range</span>
                      <p className="mt-1 font-semibold">
                        {detection.range_m ?? '—'} m
                      </p>
                    </div>

                    <div>
                      <span className="text-slate-500">Ground Range</span>
                      <p className="mt-1 font-semibold">
                        {detection.ground_range_m ?? '—'} m
                      </p>
                    </div>

                    <div>
                      <span className="text-slate-500">Target Latitude</span>
                      <p className="mt-1 font-semibold">
                        {detection.target_latitude ?? '—'}
                      </p>
                    </div>

                    <div>
                      <span className="text-slate-500">Target Longitude</span>
                      <p className="mt-1 font-semibold">
                        {detection.target_longitude ?? '—'}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 rounded-lg bg-slate-950 p-4 text-sm">
                    <p className="text-slate-500">Bounding Box</p>
                    <p className="mt-1 font-mono text-slate-300">
                      x1: {detection.bbox.x1.toFixed(2)} ·{' '}
                      y1: {detection.bbox.y1.toFixed(2)} ·{' '}
                      x2: {detection.bbox.x2.toFixed(2)} ·{' '}
                      y2: {detection.bbox.y2.toFixed(2)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  )
}

export default ScanDetail
