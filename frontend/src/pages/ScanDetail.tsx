import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { getScan } from '../services/api'
import type { ScanResult } from '../types/api'

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
        Loading scan investigation...
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
        <p className="text-sm font-medium uppercase tracking-[0.25em] text-cyan-400">
          DRISHTI • SCAN INVESTIGATION
        </p>

        <h1 className="mt-2 text-3xl font-bold">
          {scan.scan_identity}
        </h1>

        <p className="mt-2 text-slate-400">
          {new Date(scan.timestamp).toLocaleString()}
        </p>

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

            <p className="mt-4 text-4xl font-bold">
              {scan.detection_count}
            </p>

            <p className="mt-1 text-sm text-slate-400">
              detected anomalies
            </p>
          </div>
        </section>

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
                    <div>
                      <p className="font-semibold">
                        Detection {index + 1}
                      </p>

                      <p className="mt-1 text-cyan-400">
                        {detection.class_name}
                      </p>
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
