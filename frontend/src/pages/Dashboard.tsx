import { useEffect, useMemo, useState } from 'react'
import { getScans } from '../services/api'
import type { ScanResult } from '../types/api'

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
                <div
                  key={scan.id}
                  className="flex flex-col gap-4 p-5 md:flex-row md:items-center md:justify-between"
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
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  )
}

export default Dashboard
