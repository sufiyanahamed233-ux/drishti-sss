import React, { useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  ChevronRight,
  ExternalLink,
  FileSpreadsheet,
  FolderUp,
  Image as ImageIcon,
  Layers,
  Loader2,
  RotateCcw,
  Trash2,
  UploadCloud,
} from 'lucide-react'
import { analyzeBatch, ApiError } from '../services/api'
import type { BatchAnalysisResult } from '../types/api'

const VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
}

function isValidImageFile(file: File): boolean {
  const name = file.name.toLowerCase()
  return VALID_IMAGE_EXTENSIONS.some((ext) => name.endsWith(ext))
}

export default function BatchUpload() {
  const navigate = useNavigate()
  const imageInputRef = useRef<HTMLInputElement>(null)
  const csvInputRef = useRef<HTMLInputElement>(null)

  const [images, setImages] = useState<File[]>([])
  const [navigationFile, setNavigationFile] = useState<File | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)

  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)
  const [result, setResult] = useState<BatchAnalysisResult | null>(null)

  // Drag and drop hover states
  const [isDraggingImages, setIsDraggingImages] = useState(false)
  const [isDraggingCsv, setIsDraggingCsv] = useState(false)

  // Handle adding images
  const handleAddImages = (files: FileList | File[]) => {
    setValidationError(null)
    setApiError(null)

    const newFiles: File[] = []
    const invalidFiles: string[] = []

    Array.from(files).forEach((file) => {
      if (isValidImageFile(file)) {
        // Prevent exact duplicates by filename and size
        const exists = images.some(
          (img) => img.name === file.name && img.size === file.size,
        )
        if (!exists) {
          newFiles.push(file)
        }
      } else {
        invalidFiles.push(file.name)
      }
    })

    if (invalidFiles.length > 0) {
      setValidationError(
        `Ignored ${invalidFiles.length} file(s) with unsupported extensions. Only .jpg, .jpeg, and .png are accepted: ${invalidFiles.slice(0, 3).join(', ')}${invalidFiles.length > 3 ? '...' : ''}`,
      )
    }

    if (newFiles.length > 0) {
      setImages((prev) => [...prev, ...newFiles])
    }
  }

  // Handle setting CSV
  const handleSetCsv = (files: FileList | File[]) => {
    setValidationError(null)
    setApiError(null)

    if (files.length === 0) return
    const file = files[0]

    const lowerName = file.name.toLowerCase()
    if (!lowerName.endsWith('.csv')) {
      setValidationError(`Expected a CSV file, but received "${file.name}".`)
      return
    }

    if (lowerName !== 'navigation.csv') {
      setValidationError(
        `The file is named "${file.name}". The system expects navigation.csv. Please ensure it is named navigation.csv.`,
      )
    }

    setNavigationFile(file)
  }

  const handleRemoveImage = (indexToRemove: number) => {
    setImages((prev) => prev.filter((_, idx) => idx !== indexToRemove))
  }

  const handleClearImages = () => {
    setImages([])
    if (imageInputRef.current) {
      imageInputRef.current.value = ''
    }
  }

  const handleRemoveCsv = () => {
    setNavigationFile(null)
    if (csvInputRef.current) {
      csvInputRef.current.value = ''
    }
  }

  // Basic client-side validation
  const validateForm = (): boolean => {
    if (images.length === 0) {
      setValidationError('Please select at least one sonar image (.jpg, .jpeg, or .png).')
      return false
    }

    if (!navigationFile) {
      setValidationError('Please select the navigation.csv file.')
      return false
    }

    if (navigationFile.name.toLowerCase() !== 'navigation.csv') {
      setValidationError(
        `The navigation file must be named "navigation.csv" (current name: "${navigationFile.name}").`,
      )
      return false
    }

    const hasInvalidImages = images.some((img) => !isValidImageFile(img))
    if (hasInvalidImages) {
      setValidationError('All scan files must be valid images (.jpg, .jpeg, or .png).')
      return false
    }

    setValidationError(null)
    return true
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) return
    if (!navigationFile) return

    setIsAnalyzing(true)
    setApiError(null)
    setResult(null)

    try {
      const data = await analyzeBatch(images, navigationFile)
      if (data.batch_id) {
        navigate(`/batches/${data.batch_id}`)
      } else {
        setResult(data)
      }
    } catch (err) {
      if (err instanceof ApiError) {
        setApiError(err.message)
      } else if (err instanceof Error) {
        setApiError(err.message)
      } else {
        setApiError('An unexpected error occurred during batch analysis.')
      }
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setImages([])
    setNavigationFile(null)
    setResult(null)
    setApiError(null)
    setValidationError(null)
    if (imageInputRef.current) imageInputRef.current.value = ''
    if (csvInputRef.current) csvInputRef.current.value = ''
  }

  const isFormReady =
    images.length > 0 &&
    navigationFile !== null &&
    navigationFile.name.toLowerCase() === 'navigation.csv' &&
    !isAnalyzing

  return (
    <main className="min-h-screen bg-slate-950 p-6 md:p-8 text-white">
      <div className="mx-auto max-w-6xl">
        {/* Navigation Breadcrumb */}
        <div className="mb-6 flex items-center justify-between">
          <Link
            to="/"
            className="inline-flex items-center gap-2 text-sm font-medium text-slate-400 transition hover:text-cyan-400"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>

          <span className="text-xs uppercase tracking-widest text-slate-500">
            FROZEN MVP • BATCH INGESTION
          </span>
        </div>

        {/* Page Header */}
        <header className="mb-8">
          <p className="mb-2 text-sm font-medium uppercase tracking-[0.25em] text-cyan-400">
            DRISHTI • INVESTIGATION BATCH
          </p>
          <h1 className="text-3xl font-bold">Batch Sonar Upload &amp; Analysis</h1>
          <p className="mt-2 text-slate-400">
            Upload investigation batches containing raw sonar image files and platform navigation
            metadata for automated AI anomaly detection and deterministic georeferencing.
          </p>
        </header>

        {/* Error notification banner */}
        {apiError && (
          <div className="mb-8 rounded-xl border border-red-500/40 bg-red-500/10 p-5 text-red-200">
            <div className="flex items-start gap-3">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
              <div className="flex-1">
                <h3 className="font-semibold text-red-300">Batch Analysis Failed</h3>
                <p className="mt-1 text-sm leading-relaxed text-red-200/90">{apiError}</p>
                <div className="mt-3 flex gap-3">
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={isAnalyzing}
                    className="inline-flex items-center gap-1.5 rounded-md bg-red-600/80 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-red-500 cursor-pointer"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Retry Submission
                  </button>
                  <button
                    type="button"
                    onClick={() => setApiError(null)}
                    className="text-xs text-red-400 underline hover:text-red-300 cursor-pointer"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Validation warning */}
        {validationError && (
          <div className="mb-8 rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 text-amber-200">
            <div className="flex items-center gap-3">
              <AlertCircle className="h-5 w-5 shrink-0 text-amber-400" />
              <p className="text-sm">{validationError}</p>
            </div>
          </div>
        )}

        {/* Success / Result View */}
        {result ? (
          <section className="space-y-8">
            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3">
                  <div className="rounded-full bg-emerald-500/20 p-2 text-emerald-400">
                    <CheckCircle2 className="h-6 w-6" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white">Batch Analysis Complete</h2>
                    <p className="text-sm text-emerald-200/80">
                      Successfully ingested and processed investigation batch scans into PostgreSQL.
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={handleReset}
                    className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-700 hover:text-white cursor-pointer"
                  >
                    <FolderUp className="h-4 w-4" />
                    Upload Another Batch
                  </button>
                  <button
                    type="button"
                    onClick={() => navigate('/')}
                    className="inline-flex items-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-semibold text-white shadow-lg shadow-cyan-950/50 transition hover:bg-cyan-500 cursor-pointer"
                  >
                    View Dashboard
                    <ChevronRight className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>

            {/* Metrics cards */}
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
                <p className="text-sm font-medium text-slate-400">Total Scans Ingested</p>
                <p className="mt-2 text-3xl font-bold text-white">{result.total_scans}</p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
                <p className="text-sm font-medium text-slate-400">Successful Scans</p>
                <p className="mt-2 text-3xl font-bold text-emerald-400">
                  {result.successful_scans}
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-900 p-5">
                <p className="text-sm font-medium text-slate-400">Total Detections Found</p>
                <p className="mt-2 text-3xl font-bold text-cyan-400">
                  {result.total_detections}
                </p>
              </div>
            </div>

            {/* Per-scan result table */}
            <div className="rounded-xl border border-slate-800 bg-slate-900 overflow-hidden">
              <div className="border-b border-slate-800 p-5 flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-white">Batch Scans Summary</h3>
                  <p className="text-xs text-slate-400">
                    Individual scan identities, detection counts, and georeference status.
                  </p>
                </div>
                <span className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300">
                  {result.scans.length} Scans
                </span>
              </div>

              <div className="divide-y divide-slate-800">
                {result.scans.map((scan) => (
                  <div
                    key={scan.id}
                    className="flex flex-col gap-3 p-4 transition-colors hover:bg-slate-800/40 sm:flex-row sm:items-center sm:justify-between"
                  >
                    <div className="flex items-center gap-3">
                      <div className="rounded-lg bg-slate-800 p-2.5 text-cyan-400">
                        <ImageIcon className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-medium text-white">{scan.scan_identity}</p>
                        <p className="text-xs text-slate-400">
                          {new Date(scan.timestamp).toLocaleString()} • Lat:{' '}
                          {scan.sonar_latitude.toFixed(5)}, Lon:{' '}
                          {scan.sonar_longitude.toFixed(5)}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-right">
                        <span className="text-xs text-slate-500">Detections</span>
                        <p className="font-semibold text-white">
                          {scan.detection_count === 0 ? (
                            <span className="text-slate-400">0 (Zero detection)</span>
                          ) : (
                            <span className="text-cyan-400">{scan.detection_count}</span>
                          )}
                        </p>
                      </div>

                      <Link
                        to={`/scans/${scan.id}`}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1.5 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white"
                      >
                        Scan Detail
                        <ExternalLink className="h-3 w-3" />
                      </Link>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        ) : (
          /* Upload & Configuration Form */
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="grid gap-6 lg:grid-cols-2">
              {/* Card 1: Sonar Images */}
              <div className="flex flex-col rounded-xl border border-slate-800 bg-slate-900 p-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-cyan-950 p-2 text-cyan-400">
                      <ImageIcon className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="font-semibold text-white">1. Sonar Images</h2>
                      <p className="text-xs text-slate-400">Accepted: .jpg, .jpeg, .png</p>
                    </div>
                  </div>
                  <span className="rounded-full bg-slate-800 px-2.5 py-0.5 text-xs font-medium text-slate-300">
                    {images.length} selected
                  </span>
                </div>

                {/* Drag and Drop Zone for Images */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault()
                    if (!isAnalyzing) setIsDraggingImages(true)
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault()
                    setIsDraggingImages(false)
                  }}
                  onDrop={(e) => {
                    e.preventDefault()
                    setIsDraggingImages(false)
                    if (isAnalyzing) return
                    if (e.dataTransfer.files) {
                      handleAddImages(e.dataTransfer.files)
                    }
                  }}
                  className={`mt-4 flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition ${
                    isDraggingImages
                      ? 'border-cyan-500 bg-cyan-950/20'
                      : 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                  }`}
                >
                  <UploadCloud className="h-9 w-9 text-slate-400" />
                  <p className="mt-2 text-sm font-medium text-slate-300">
                    Drag and drop sonar scan images here
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Select multiple files corresponding to rows in navigation.csv
                  </p>

                  <input
                    ref={imageInputRef}
                    type="file"
                    multiple
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    disabled={isAnalyzing}
                    onChange={(e) => {
                      if (e.target.files) handleAddImages(e.target.files)
                    }}
                    className="hidden"
                    id="sonar-images-input"
                  />
                  <label
                    htmlFor="sonar-images-input"
                    className="mt-4 inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white"
                  >
                    Browse Images
                  </label>
                </div>

                {/* Selected Images List */}
                <div className="mt-4 flex-1">
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span>Selected Files</span>
                    {images.length > 0 && !isAnalyzing && (
                      <button
                        type="button"
                        onClick={handleClearImages}
                        className="text-xs text-slate-500 hover:text-red-400 cursor-pointer"
                      >
                        Clear All
                      </button>
                    )}
                  </div>

                  {images.length === 0 ? (
                    <div className="mt-2 rounded-lg border border-dashed border-slate-800/80 p-6 text-center text-xs text-slate-500">
                      No scan images selected yet.
                    </div>
                  ) : (
                    <div className="mt-2 max-h-56 overflow-y-auto space-y-1.5 pr-1">
                      {images.map((img, idx) => (
                        <div
                          key={`${img.name}-${img.size}-${idx}`}
                          className="flex items-center justify-between rounded-lg bg-slate-950/60 px-3 py-2 text-xs text-slate-300"
                        >
                          <div className="flex items-center gap-2 truncate">
                            <span className="text-slate-500">#{idx + 1}</span>
                            <span className="truncate font-medium text-slate-200">{img.name}</span>
                            <span className="text-slate-500">({formatBytes(img.size)})</span>
                          </div>
                          {!isAnalyzing && (
                            <button
                              type="button"
                              onClick={() => handleRemoveImage(idx)}
                              title={`Remove ${img.name}`}
                              className="text-slate-500 transition hover:text-red-400 cursor-pointer"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Card 2: navigation.csv */}
              <div className="flex flex-col rounded-xl border border-slate-800 bg-slate-900 p-6">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-cyan-950 p-2 text-cyan-400">
                      <FileSpreadsheet className="h-5 w-5" />
                    </div>
                    <div>
                      <h2 className="font-semibold text-white">2. Navigation Metadata</h2>
                      <p className="text-xs text-slate-400">Exactly one navigation.csv</p>
                    </div>
                  </div>
                  <span
                    className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                      navigationFile
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {navigationFile ? '1 Selected' : 'Required'}
                  </span>
                </div>

                {/* Drag and Drop Zone for CSV */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault()
                    if (!isAnalyzing) setIsDraggingCsv(true)
                  }}
                  onDragLeave={(e) => {
                    e.preventDefault()
                    setIsDraggingCsv(false)
                  }}
                  onDrop={(e) => {
                    e.preventDefault()
                    setIsDraggingCsv(false)
                    if (isAnalyzing) return
                    if (e.dataTransfer.files) {
                      handleSetCsv(e.dataTransfer.files)
                    }
                  }}
                  className={`mt-4 flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 text-center transition ${
                    isDraggingCsv
                      ? 'border-cyan-500 bg-cyan-950/20'
                      : 'border-slate-800 bg-slate-950/40 hover:border-slate-700'
                  }`}
                >
                  <FileSpreadsheet className="h-9 w-9 text-slate-400" />
                  <p className="mt-2 text-sm font-medium text-slate-300">
                    Drag and drop navigation.csv here
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    File must be named navigation.csv
                  </p>

                  <input
                    ref={csvInputRef}
                    type="file"
                    accept=".csv,text/csv"
                    disabled={isAnalyzing}
                    onChange={(e) => {
                      if (e.target.files) handleSetCsv(e.target.files)
                    }}
                    className="hidden"
                    id="navigation-csv-input"
                  />
                  <label
                    htmlFor="navigation-csv-input"
                    className="mt-4 inline-flex cursor-pointer items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-700 hover:text-white"
                  >
                    {navigationFile ? 'Replace navigation.csv' : 'Browse navigation.csv'}
                  </label>
                </div>

                {/* Selected CSV Display */}
                <div className="mt-4 flex-1">
                  <div className="text-xs text-slate-400">Selected Navigation File</div>

                  {navigationFile ? (
                    <div className="mt-2 flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-200">
                      <div className="flex items-center gap-2.5 truncate">
                        <FileSpreadsheet className="h-4 w-4 shrink-0 text-cyan-400" />
                        <span className="font-semibold text-white truncate">
                          {navigationFile.name}
                        </span>
                        <span className="text-slate-400">
                          ({formatBytes(navigationFile.size)})
                        </span>
                      </div>
                      {!isAnalyzing && (
                        <button
                          type="button"
                          onClick={handleRemoveCsv}
                          title="Remove navigation file"
                          className="text-slate-500 transition hover:text-red-400 cursor-pointer"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="mt-2 rounded-lg border border-dashed border-slate-800/80 p-6 text-center text-xs text-slate-500">
                      No navigation.csv file selected.
                    </div>
                  )}

                  {/* Architecture reference box */}
                  <div className="mt-4 rounded-lg bg-slate-950/40 p-3 text-xs text-slate-400 border border-slate-800/60">
                    <p className="font-semibold text-slate-300">Investigation Batch Layout:</p>
                    <pre className="mt-1 font-mono text-[11px] text-cyan-400">
{`investigation_batch/
├── scans/ (Multiple .jpg / .png)
└── navigation.csv (1-to-1 scan mapping)`}
                    </pre>
                  </div>
                </div>
              </div>
            </div>

            {/* Submission Actions */}
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h3 className="font-semibold text-white">Ready to Analyze</h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {images.length > 0 && navigationFile
                      ? `Batch prepared: ${images.length} scan image${images.length !== 1 ? 's' : ''} + ${navigationFile.name}`
                      : 'Please select both scan images and navigation.csv to proceed.'}
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  <button
                    type="submit"
                    disabled={!isFormReady}
                    className={`inline-flex items-center justify-center gap-2 rounded-lg px-6 py-3 text-sm font-semibold transition ${
                      isFormReady
                        ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-950/50 hover:bg-cyan-500 cursor-pointer'
                        : 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-800'
                    }`}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin text-cyan-300" />
                        <span>Analyzing batch...</span>
                      </>
                    ) : (
                      <>
                        <Layers className="h-4 w-4" />
                        <span>Analyze Batch</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* In-progress helper note */}
              {isAnalyzing && (
                <div className="mt-4 flex items-center gap-3 rounded-lg border border-cyan-500/20 bg-cyan-950/30 p-3 text-xs text-cyan-300">
                  <Loader2 className="h-4 w-4 animate-spin shrink-0 text-cyan-400" />
                  <span>
                    Executing investigation batch analysis: uploading files, performing YOLO object detection, georeferencing coordinates, and persisting records into PostgreSQL...
                  </span>
                </div>
              )}
            </div>
          </form>
        )}
      </div>
    </main>
  )
}
