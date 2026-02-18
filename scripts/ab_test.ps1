param(
    [string]$Switchboard = "configs/sr_switchboard.yaml",
    [string]$Input = "./data/gtav/2.png",
    [string]$OutputRoot = "",
    [int]$RepeatInput = 12,
    [int]$PreviewFrames = 4,
    [int]$MontageFrameIndex = 0,
    [switch]$BuildMontage,
    [switch]$BuildVideo
)

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = "./outputs/ab_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$cases = @(
    @{ Name = "baseline_legacy_trt"; Upsampler = "legacy_trt"; Post = @() },
    @{ Name = "legacy_trt_desra"; Upsampler = "legacy_trt"; Post = @("desra_blend") },
    @{ Name = "legacy_trt_vsr"; Upsampler = "legacy_trt"; Post = @("vsr_realbasicvsr") },
    @{ Name = "swinir_x2_trt"; Upsampler = "swinir_realsr_x2_trt"; Post = @() },
    @{ Name = "swinir_x2_trt_desra"; Upsampler = "swinir_realsr_x2_trt"; Post = @("desra_blend") },
    @{ Name = "hat_x2_trt"; Upsampler = "hat_x2_trt"; Post = @() },
    @{ Name = "rgt_x2_trt"; Upsampler = "rgt_x2_trt"; Post = @() }
)

$results = @()
$montageRuns = @()

foreach ($case in $cases) {
    $runName = $case.Name
    $runDir = Join-Path $OutputRoot $runName

    $argsList = @(
        "tools/run_pipeline.py",
        "--switchboard", $Switchboard,
        "--upsampler", $case.Upsampler,
        "--input", $Input,
        "--repeat_input", $RepeatInput.ToString(),
        "--out_dir", $runDir,
        "--make_preview_count", $PreviewFrames.ToString()
    )

    foreach ($postStage in $case.Post) {
        $argsList += @("--post", $postStage)
    }

    Write-Host ""
    Write-Host "=== Running $runName ==="
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    & python @argsList
    $exitCode = $LASTEXITCODE
    $timer.Stop()

    $metricsPath = Join-Path $runDir "metrics.json"
    $avgMs = $null
    $frameCount = $null
    $finalFramesDir = $null
    if (Test-Path $metricsPath) {
        $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
        $avgMs = $metrics.upsample_result.metrics.avg_ms_per_frame
        $frameCount = $metrics.upsample_result.metrics.frame_count
        $finalFramesDir = $metrics.final_frames_dir
    }

    $gpuMemSnapshotMb = $null
    try {
        $gpuMemSnapshotMb = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim()
    } catch {
        $gpuMemSnapshotMb = $null
    }

    $results += [PSCustomObject]@{
        run_name = $runName
        upsampler = $case.Upsampler
        post = ($case.Post -join "+")
        status = if ($exitCode -eq 0) { "ok" } else { "failed" }
        total_seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 3)
        avg_ms_per_frame = $avgMs
        frame_count = $frameCount
        gpu_mem_snapshot_mb = $gpuMemSnapshotMb
        run_dir = $runDir
    }

    if ($exitCode -eq 0 -and $finalFramesDir) {
        $montageRuns += "{0}={1}" -f $runName, $finalFramesDir
        if ($BuildVideo) {
            try {
                $videoOut = Join-Path $runDir "preview.mp4"
                & ffmpeg -y -framerate 10 -i (Join-Path $finalFramesDir "frame_%06d.png") -c:v libx264 -pix_fmt yuv420p $videoOut | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "ffmpeg failed for $runName"
                }
            } catch {
                Write-Warning "ffmpeg not available; skipping mp4 preview for $runName"
            }
        }
    }
}

$csvPath = Join-Path $OutputRoot "ab_results.csv"
$jsonPath = Join-Path $OutputRoot "ab_results.json"
$results | Export-Csv -Path $csvPath -NoTypeInformation
$results | ConvertTo-Json -Depth 6 | Set-Content -Path $jsonPath -Encoding UTF8

Write-Host ""
Write-Host "Saved results:"
Write-Host " - $csvPath"
Write-Host " - $jsonPath"

if ($BuildMontage -and $montageRuns.Count -gt 0) {
    for ($i = 0; $i -lt $PreviewFrames; $i++) {
        $frameIndex = $MontageFrameIndex + $i
        $montageOut = Join-Path $OutputRoot ("ab_montage_frame_{0:D4}.png" -f $frameIndex)
        $montageArgs = @("tools/make_ab_montage.py")
        foreach ($runSpec in $montageRuns) {
            $montageArgs += @("--run", $runSpec)
        }
        $montageArgs += @("--frame_index", $frameIndex.ToString(), "--out", $montageOut)

        & python @montageArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Saved montage: $montageOut"
        } else {
            Write-Warning "Montage generation failed for frame index $frameIndex."
        }
    }
}
