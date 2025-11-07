#requires -Version 7.0
<#
    RunBot.ps1
    Launcher for the AI-Trading-Bot stack on Windows.

    Usage:
      .\RunBot.ps1              # start services and GUI
      .\RunBot.ps1 -DebugMode   # start with verbose logging and live tails
      .\RunBot.ps1 -Stop        # stop previously started services
#>

[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$DebugMode
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding  = [System.Text.Encoding]::UTF8

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
$PythonExe       = "$PSScriptRoot\.venv\Scripts\python.exe"
$UseUV           = $false
$PREFECT_UI      = "http://127.0.0.1:4200"
$PREFECT_API_URL = "http://127.0.0.1:4200/api"
$WorkPool        = "bitmex"
$UseDockerAgent  = $false
$GuiMode         = "flask" # flask | streamlit | fastapi | gradio
$StreamlitEntry  = "ui/app.py"
$FlaskEntry      = "run.py"
$FastAPIApp      = "orchestration.api:app"
$GradioEntry     = "ui/gradio_app.py"
$LogsDir         = "$PSScriptRoot\logs"
$EnvFile         = "$PSScriptRoot\.env"
$Requirements    = "$PSScriptRoot\requirements.txt"
$StateFile       = Join-Path $LogsDir "runbot-state.json"
$ManageKafkaStack = $true
$KafkaComposeFile = Join-Path $PSScriptRoot "docker\docker-compose.kafka.yml"
$KafkaProjectName = "trading-bot"
$KafkaNetworkName = "kafka-net"

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
$script:ProcessRegistry   = @()
$script:TailJobs          = @()
$script:MonitorJobs       = @()
$script:ServerStarted     = $false
$script:ShutdownRequested = $false
$script:CleanupComplete   = $false
$script:PrefectExe        = "$PSScriptRoot\.venv\Scripts\prefect.exe"
$script:PipRepairAttempted = $false

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
function Write-Log {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ValidateSet("INFO", "WARN", "ERROR", "SUCCESS", "DEBUG")][string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $color = switch ($Level) {
        "INFO"    { "Cyan" }
        "WARN"    { "Yellow" }
        "ERROR"   { "Red" }
        "SUCCESS" { "Green" }
        "DEBUG"   { "DarkGray" }
    }
    Write-Host ("[{0}][{1}] {2}" -f $timestamp, $Level, $Message) -ForegroundColor $color
}

function Test-Port {
    param(
        [Parameter(Mandatory = $true)][string]$TargetHost,
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutMs = 2000
    )
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect($TargetHost, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMs)) {
            $client.Close()
            return $false
        }
        $client.EndConnect($async)
        $client.Close()
        return $true
    } catch {
        try { $client.Close() } catch { }
        return $false
    }
}

function Wait-Http-200 {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 60,
        [int]$IntervalSeconds = 3
    )
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.AllowAutoRedirect = $true
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(10)
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
            try {
                $response = $client.GetAsync($Url).GetAwaiter().GetResult()
                if ($response.IsSuccessStatusCode) {
                    return $true
                }
            } catch {
                Start-Sleep -Seconds $IntervalSeconds
                continue
            }
            Start-Sleep -Seconds $IntervalSeconds
        }
        return $false
    } finally {
        $client.Dispose()
        $handler.Dispose()
    }
}

function Ensure-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string]$FriendlyName
    )
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "$FriendlyName not found on PATH. Please install it before continuing."
    }
}

function Invoke-FrontendBundle {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$BundlePath
    )

    if (-not (Test-Path $SourcePath)) {
        Write-Log "Frontend source not found at $SourcePath; skipping bundle build." "WARN"
        return
    }

    $mode = ($env:RUNBOT_BUILD_BUNDLE ?? "force").ToString().ToLowerInvariant()
    if ($mode -in @("0", "false", "skip", "off")) {
        Write-Log "RUNBOT_BUILD_BUNDLE=$mode -> skipping React bundle build." "DEBUG"
        return
    }

    $shouldBuild = $true
    if ($mode -in @("", "auto", "changed")) {
        $shouldBuild = $false
        if (-not (Test-Path $BundlePath)) {
            $shouldBuild = $true
        } elseif ((Get-Item $BundlePath).LastWriteTimeUtc -lt (Get-Item $SourcePath).LastWriteTimeUtc) {
            $shouldBuild = $true
        }
    }

    if (-not $shouldBuild) {
        Write-Log "Frontend bundle up-to-date; skipping build." "DEBUG"
        return
    }

    $npx = Get-Command "npx" -ErrorAction SilentlyContinue
    if (-not $npx) {
        Write-Log "npx not found on PATH; cannot refresh React bundle." "WARN"
        return
    }

    $bundleDir = Split-Path -Parent $BundlePath
    if (-not (Test-Path $bundleDir)) {
        New-Item -ItemType Directory -Path $bundleDir -Force | Out-Null
    }

    $args = @(
        "esbuild",
        $SourcePath,
        "--loader:.jsx=jsx",
        "--outfile=$BundlePath",
        "--format=iife",
        "--minify"
    )

    Write-Log "Building React bundle via esbuild..." "INFO"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $npx.Source
    $psi.WorkingDirectory = $PSScriptRoot
    foreach ($arg in $args) {
        [void]$psi.ArgumentList.Add($arg)
    }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false

    $process = [System.Diagnostics.Process]::Start($psi)
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    if ($stdout) { Write-Log $stdout.Trim() "DEBUG" }
    if ($stderr) { Write-Log $stderr.Trim() ($(if ($process.ExitCode -eq 0) { "DEBUG" } else { "ERROR" })) }

    if ($process.ExitCode -ne 0) {
        throw "esbuild exited with code $($process.ExitCode); see logs above for details."
    }

    Write-Log "React bundle updated at $BundlePath" "SUCCESS"
}

function Ensure-UiVendorAssets {
    $vendorDir = Join-Path $PSScriptRoot "tradingbotui\static\vendor"
    New-Item -ItemType Directory -Path $vendorDir -Force | Out-Null

    $mode = ($env:RUNBOT_REFRESH_VENDOR ?? "auto").ToString().ToLowerInvariant()
    $force = $mode -in @("force", "1", "true", "yes", "refresh")

    $assets = @(
        @{ Name = "react.production.min.js"; Url = "https://unpkg.com/react@18/umd/react.production.min.js" },
        @{ Name = "react-dom.production.min.js"; Url = "https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" },
        @{ Name = "dayjs.min.js"; Url = "https://unpkg.com/dayjs@1/dayjs.min.js" },
        @{ Name = "dayjs-plugin-relativeTime.js"; Url = "https://unpkg.com/dayjs@1/plugin/relativeTime.js" },
        @{ Name = "chart.umd.min.js"; Url = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js" }
    )

    foreach ($asset in $assets) {
        $targetPath = Join-Path $vendorDir $asset.Name
        $shouldDownload = $force -or -not (Test-Path $targetPath)
        if (-not $shouldDownload) {
            continue
        }
        Write-Log ("Fetching UI asset {0}..." -f $asset.Name) "INFO"
        $tempFile = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
        try {
            Invoke-WebRequest -Uri $asset.Url -OutFile $tempFile -UseBasicParsing -TimeoutSec 120 | Out-Null
            Move-Item -Path $tempFile -Destination $targetPath -Force
            Write-Log ("Updated {0}" -f $asset.Name) "DEBUG"
        } catch {
            if (Test-Path $tempFile) { Remove-Item $tempFile -Force -ErrorAction SilentlyContinue }
            $message = "Failed to download {0}: {1}" -f $asset.Name, $_.Exception.Message
            if (-not (Test-Path $targetPath)) {
                throw $message
            } else {
                Write-Log "$message (using existing copy)" "WARN"
            }
        }
    }
}

function Ensure-Process {
    param(
        [Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process,
        [string]$Name = "process"
    )
    if ($Process.HasExited) {
        Write-Log "$Name exited with code $($Process.ExitCode)" "ERROR"
        return $false
    }
    return $true
}

function Start-Logged {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string]$WorkingDirectory = $PSScriptRoot
    )
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdoutPath = Join-Path $LogsDir ("{0}-{1}.out.log" -f $Name, $timestamp)
    $stderrPath = Join-Path $LogsDir ("{0}-{1}.err.log" -f $Name, $timestamp)
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
    New-Item -ItemType File -Path $stdoutPath -Force | Out-Null
    New-Item -ItemType File -Path $stderrPath -Force | Out-Null

    Write-Log "Starting $Name -> $FilePath $($ArgumentList -join ' ')" "INFO"
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -NoNewWindow -PassThru
    Write-Log "$Name running (PID $($process.Id))" "SUCCESS"
    return [pscustomobject]@{
        Name      = $Name
        Process   = $process
        StdoutLog = $stdoutPath
        StderrLog = $stderrPath
    }
}

function Register-ManagedProcess {
    param(
        [Parameter(Mandatory = $true)]$ProcessInfo,
        [bool]$StartedByScript = $true
    )
    if ($null -eq $ProcessInfo) { return }
    $entry = [pscustomobject]@{
        Name             = $ProcessInfo.Name
        Process          = $ProcessInfo.Process
        StdoutLog        = $ProcessInfo.StdoutLog
        StderrLog        = $ProcessInfo.StderrLog
        StartedByScript  = $StartedByScript
    }
    $script:ProcessRegistry += $entry
    return $entry
}

function Load-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return }
    Write-Log "Loading environment from $Path" "INFO"
    foreach ($line in Get-Content $Path) {
        if ($line -match '^\s*#' -or $line -match '^\s*$') { continue }
        $pair = $line -split '=', 2
        if ($pair.Count -ne 2) { continue }
        $key = $pair[0].Trim()
        $value = $pair[1].Trim()
        [Environment]::SetEnvironmentVariable($key, $value, 'Process')
    }
}

function Invoke-ProcessWithInput {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [string[]]$InputLines = @(),
        [string]$WorkingDirectory = $PSScriptRoot,
        [string]$Description = ""
    )
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $FilePath
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    foreach ($arg in $ArgumentList) {
        [void]$psi.ArgumentList.Add($arg)
    }

    $process = [System.Diagnostics.Process]::Start($psi)
    try {
        if ($InputLines.Count -gt 0) {
            foreach ($line in $InputLines) {
                $process.StandardInput.WriteLine($line)
            }
        }
        $process.StandardInput.Close()
        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        if ($Description) {
            Write-Log "$Description exited with code $($process.ExitCode)" "DEBUG"
        }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout   = $stdout
            Stderr   = $stderr
        }
    } finally {
        $process.Dispose()
    }
}

function Ensure-KafkaNetwork {
    if (-not $ManageKafkaStack) { return }
    try {
        & docker network inspect $KafkaNetworkName *> $null 2>&1
        if ($LASTEXITCODE -eq 0) { return }
        Write-Log "Creating Docker network '$KafkaNetworkName'..." "INFO"
        & docker network create $KafkaNetworkName *> $null 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Failed to create Docker network '$KafkaNetworkName' (exit $LASTEXITCODE)." "WARN"
        }
    } catch {
        Write-Log "Docker network check failed: $($_.Exception.Message)" "WARN"
    }
}

function Remove-StaleKafkaContainers {
    if (-not $ManageKafkaStack) { return }
    if (-not (Test-Path $KafkaComposeFile)) { return }
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) { return }
    $args = @("compose", "-p", $KafkaProjectName, "-f", $KafkaComposeFile, "down", "--remove-orphans")
    try {
        $result = Invoke-ProcessWithInput -FilePath "docker" -ArgumentList $args -Description "docker compose down (pre-clean Kafka stack)"
    } catch {
        Write-Log "Pre-clean Kafka stack removal failed: $($_.Exception.Message)" "WARN"
        return
    }
    if ($result.ExitCode -eq 0) {
        Write-Log "Cleared previous Kafka/ZooKeeper containers (project '$KafkaProjectName')." "INFO"
    } else {
        $msg = $result.Stderr.Trim()
        if (-not $msg) { $msg = $result.Stdout.Trim() }
        if (-not $msg) { $msg = "exit $($result.ExitCode)" }
        Write-Log "Pre-clean Kafka stack removal returned exit $($result.ExitCode): $msg" "WARN"
    }
}

function Start-KafkaStack {
    if (-not $ManageKafkaStack) { return }
    if (-not (Test-Path $KafkaComposeFile)) {
        Write-Log "Kafka compose file not found at $KafkaComposeFile; skipping broker startup." "WARN"
        return
    }
    try {
        Ensure-Command -Command "docker" -FriendlyName "Docker CLI"
    } catch {
        Write-Log "Docker CLI not found; skipping Kafka/ZooKeeper stack startup. Set RUNBOT_SKIP_KAFKA=1 to hide this warning." "WARN"
        $script:ManageKafkaStack = $false
        return
    }
    Remove-StaleKafkaContainers
    Ensure-KafkaNetwork
    $args = @("compose", "-p", $KafkaProjectName, "-f", $KafkaComposeFile, "up", "-d", "--remove-orphans")
    try {
        $result = Invoke-ProcessWithInput -FilePath "docker" -ArgumentList $args -Description "docker compose up (Kafka stack)"
    } catch {
        Write-Log "Kafka/ZooKeeper launch threw an exception: $($_.Exception.Message). Continuing without Kafka." "WARN"
        return
    }
    if ($result.ExitCode -ne 0) {
        $msg = $result.Stderr.Trim()
        if (-not $msg) { $msg = $result.Stdout.Trim() }
        if (-not $msg) { $msg = "exit $($result.ExitCode)" }
        Write-Log "Kafka/ZooKeeper launch encountered an error ($msg). Continuing without Kafka." "WARN"
        return
    }
    Write-Log "Kafka/ZooKeeper stack running (project '$KafkaProjectName')." "SUCCESS"
}

function Stop-KafkaStack {
    if (-not $ManageKafkaStack) { return }
    if (-not (Test-Path $KafkaComposeFile)) { return }
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (-not $dockerCmd) { return }
    $args = @("compose", "-p", $KafkaProjectName, "-f", $KafkaComposeFile, "down")
    try {
        $result = Invoke-ProcessWithInput -FilePath "docker" -ArgumentList $args -Description "docker compose down (Kafka stack)"
    } catch {
        Write-Log "Kafka stack shutdown threw an exception: $($_.Exception.Message)" "WARN"
        return
    }
    if ($result.ExitCode -ne 0) {
        $msg = $result.Stderr.Trim()
        if (-not $msg) { $msg = $result.Stdout.Trim() }
        if (-not $msg) { $msg = "exit $($result.ExitCode)" }
        Write-Log "Kafka stack shutdown returned exit $($result.ExitCode): $msg" "WARN"
    } else {
        Write-Log "Kafka/ZooKeeper stack stopped." "INFO"
    }
}

function Get-SitePackagesPath {
    $scriptsDir = Split-Path $PythonExe -Parent
    $venvRoot = Split-Path $scriptsDir -Parent
    return Join-Path $venvRoot "Lib\site-packages"
}

function Repair-Pip {
    param(
        [bool]$FullReset = $false,
        [string]$Reason = ""
    )
    if ($script:PipRepairAttempted) {
        return $false
    }
    $script:PipRepairAttempted = $true
    if ($Reason) {
        Write-Log "Attempting to repair pip installation ($Reason)..." "WARN"
    } else {
        Write-Log "Attempting to repair pip installation..." "WARN"
    }
    $sitePackages = Get-SitePackagesPath
    if (-not (Test-Path $sitePackages)) {
        Write-Log "Site-packages path $sitePackages not found; skipping repair." "WARN"
        return $false
    }
    if ($FullReset) {
        Write-Log "Performing full pip/site-packages reset..." "WARN"
        try {
            Remove-Item -Path $sitePackages -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Log "Failed to remove site-packages: $($_.Exception.Message)" "WARN"
        }
    } else {
        Get-ChildItem -Path $sitePackages -Filter "pip*" -Force -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction Stop
        } catch {
            Write-Log "Failed to remove $($_.FullName): $($_.Exception.Message)" "WARN"
        }
    }
    }
    if (-not (Test-Path $sitePackages)) {
        New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
    }
    & $PythonExe -m ensurepip --upgrade
    if ($LASTEXITCODE -eq 0) {
        Write-Log "pip repair completed." "SUCCESS"
        return $true
    } else {
        Write-Log "ensurepip repair step failed (exit $LASTEXITCODE)." "WARN"
        return $false
    }
}

function Invoke-Pip {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )
    $attempt = 0
    while ($attempt -lt 2) {
        $attempt++
        $output = & $PythonExe -m pip @Arguments 2>&1
        $exit = $LASTEXITCODE
        if ($exit -eq 0) {
            return ($output | Out-String)
        }
        $outputString = ($output | Out-String)
        Write-Log "$Description failed (attempt $attempt, exit $exit)." "WARN"
        if (-not $script:PipRepairAttempted -and $outputString -match "RequirementInformation") {
            if (Repair-Pip -Reason "pip resolver import error detected") {
                continue
            }
        } elseif (-not $script:PipRepairAttempted -and $outputString -match "No module named pip") {
            if (Repair-Pip -FullReset $true -Reason "pip module missing") {
                continue
            }
        }
        throw "pip command failed: $Description`n$outputString"
    }
}

function Start-LogTail {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Path
    )
    if (-not (Test-Path $Path)) {
        New-Item -ItemType File -Path $Path -Force | Out-Null
    }
    $job = Start-Job -Name "tail-$Name" -ArgumentList $Name, $Path -ScriptBlock {
        param($InnerName, $InnerPath)
        while (-not (Test-Path $InnerPath)) { Start-Sleep -Seconds 1 }
        Write-Host ("[TAIL][{0}] watching {1}" -f $InnerName, $InnerPath) -ForegroundColor DarkCyan
        Get-Content -Path $InnerPath -Encoding UTF8 -Tail 20 -Wait | ForEach-Object {
            Write-Host ("[{0}] {1}" -f $InnerName, $_) -ForegroundColor DarkGray
        }
    }
    $script:TailJobs += $job
}

function Stop-LogTails {
    foreach ($job in $script:TailJobs) {
        try {
            if ($job -and $job.State -in @('Running','NotStarted')) {
                Stop-Job -Job $job -Force -ErrorAction SilentlyContinue
            }
        } catch { }
        try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch { }
    }
    $script:TailJobs = @()
}

function Start-LogMonitor {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Filter = "404|ERROR|Traceback|Exception"
    )
    if (-not (Test-Path $Path)) {
        Write-Log "Monitor $Name cannot watch $Path (file missing)." "DEBUG"
        return
    }
    Write-Log "Starting monitor $Name on $Path" "DEBUG"
    $job = Start-Job -Name "monitor-$Name" -ArgumentList $Name, $Path, $Filter -ScriptBlock {
        param($InnerName, $InnerPath, $InnerFilter)
        Write-Host ("[MONITOR][{0}] {1}" -f $InnerName, $InnerPath) -ForegroundColor Yellow
        Get-Content -Path $InnerPath -Encoding UTF8 -Tail 0 -Wait | ForEach-Object {
            if ($_ -match $InnerFilter) {
                Write-Host ("[MONITOR][{0}] {1}" -f $InnerName, $_) -ForegroundColor Yellow
            }
        }
    }
    $script:MonitorJobs += $job
}

function Stop-LogMonitors {
    foreach ($job in $script:MonitorJobs) {
        try {
            if ($job -and $job.State -in @('Running','NotStarted')) {
                Stop-Job -Job $job -Force -ErrorAction SilentlyContinue
            }
        } catch { }
        try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch { }
    }
    $script:MonitorJobs = @()
}

function Save-RunState {
    $entries = @()
    foreach ($proc in $script:ProcessRegistry) {
        if ($proc.StartedByScript -and $proc.Process -and -not $proc.Process.HasExited) {
            $entries += @{
                Name      = $proc.Name
                Id        = $proc.Process.Id
                StdoutLog = $proc.StdoutLog
                StderrLog = $proc.StderrLog
            }
        }
    }
    if ($entries.Count -eq 0) {
        if (Test-Path $StateFile) {
            Remove-Item -Path $StateFile -Force -ErrorAction SilentlyContinue
        }
        return
    }
    $state = @{
        Processes = $entries
        PrefectUi = $PREFECT_UI
        GuiMode   = $GuiMode
        Timestamp = (Get-Date).ToString("o")
    }
    $json = $state | ConvertTo-Json -Depth 5
    New-Item -ItemType Directory -Path (Split-Path $StateFile -Parent) -Force | Out-Null
    Set-Content -Path $StateFile -Value $json -Encoding UTF8
}

function Remove-RunState {
    if (Test-Path $StateFile) {
        Remove-Item -Path $StateFile -Force -ErrorAction SilentlyContinue
    }
}

function Stop-RunBotCurrent {
    if ($script:CleanupComplete) { return }
    $script:CleanupComplete = $true
    Write-Log "Stopping all managed processes..." "WARN"
    Stop-LogTails
    foreach ($entry in $script:ProcessRegistry) {
        if ($null -eq $entry.Process) { continue }
        if ($entry.StartedByScript -and -not $entry.Process.HasExited) {
            try {
                Write-Log "Stopping $($entry.Name) (PID $($entry.Process.Id))" "INFO"
                Stop-Process -Id $entry.Process.Id -Force -ErrorAction Stop
            } catch {
                Write-Log "Failed to stop $($entry.Name): $($_.Exception.Message)" "WARN"
            }
        }
    }
    try {
        Stop-KafkaStack
    } catch {
        Write-Log "Kafka stack shutdown encountered a problem: $($_.Exception.Message)" "WARN"
    }
    Remove-RunState
    Stop-LogMonitors
    Write-Log "RunBot stop complete." "SUCCESS"
}

function Stop-RunBotFromState {
    if (-not (Test-Path $StateFile)) {
        Write-Log "No RunBot state file found; nothing to stop." "WARN"
        return
    }
    $stateJson = Get-Content -Path $StateFile -Raw -ErrorAction SilentlyContinue
    if (-not $stateJson) {
        Remove-RunState
        Write-Log "State file was empty. Cleared." "WARN"
        return
    }
    $state = $stateJson | ConvertFrom-Json
    Write-Log "Stopping processes recorded in state file..." "WARN"
    foreach ($proc in $state.Processes) {
        try {
            $existing = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
            if ($existing) {
                Write-Log "Stopping $($proc.Name) (PID $($proc.Id))" "INFO"
                Stop-Process -Id $proc.Id -Force -ErrorAction Stop
            } else {
                Write-Log "$($proc.Name) (PID $($proc.Id)) not running." "WARN"
            }
        } catch {
            Write-Log "Failed to stop $($proc.Name): $($_.Exception.Message)" "WARN"
        }
    }
    try {
        Stop-KafkaStack
    } catch {
        Write-Log "Kafka stack shutdown encountered a problem: $($_.Exception.Message)" "WARN"
    }
    Remove-RunState
    Stop-LogMonitors
    Write-Log "Stop request processed." "SUCCESS"
}

function Initialize-Python {
    if ($UseUV) {
        Ensure-Command -Command "uv" -FriendlyName "uv"
        if (-not (Test-Path $PythonExe)) {
            Write-Log "Creating virtual environment with uv..." "INFO"
            Push-Location $PSScriptRoot
            try {
                & uv venv .venv --python 3.11
                if ($LASTEXITCODE -ne 0) { throw "uv venv failed (exit $LASTEXITCODE)" }
            } finally {
                Pop-Location
            }
        }
        if (Test-Path $Requirements) {
            Write-Log "Installing requirements via uv..." "INFO"
            Push-Location $PSScriptRoot
            try {
                & uv pip install -r $Requirements
                if ($LASTEXITCODE -ne 0) { throw "uv pip install failed (exit $LASTEXITCODE)" }
            } finally {
                Pop-Location
            }
        }
    } else {
        if (-not (Test-Path $PythonExe)) {
            Write-Log "Creating virtual environment with py -3.11..." "INFO"
            Ensure-Command -Command "py" -FriendlyName "Python launcher (py)"
            Push-Location $PSScriptRoot
            try {
                & py -3.11 -m venv .venv
                if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
            } finally {
                Pop-Location
            }
        }
        Push-Location -Path $PSScriptRoot
        try {
            Write-Log "Upgrading pip..." "INFO"
            try {
                Invoke-Pip -Arguments @("install", "--upgrade", "pip", "--disable-pip-version-check") -Description "pip upgrade"
            } catch {
                Write-Log "pip upgrade failed: $($_.Exception.Message). Attempting ensurepip fallback..." "WARN"
                try {
                    & $PythonExe -m ensurepip --upgrade
                    if ($LASTEXITCODE -ne 0) {
                        Write-Log "ensurepip fallback exited with $LASTEXITCODE; continuing with existing pip." "WARN"
                    } else {
                        Write-Log "ensurepip fallback completed." "SUCCESS"
                    }
                } catch {
                    Write-Log "ensurepip fallback failed: $($_.Exception.Message). Continuing with existing pip." "WARN"
                }
            }
            if (Test-Path $Requirements) {
                Write-Log "Installing dependencies from $Requirements..." "INFO"
                try {
                    Invoke-Pip -Arguments @("install", "-r", $Requirements) -Description "dependency installation"
                } catch {
                    throw "Dependency installation failed: $($_.Exception.Message)"
                }
            } else {
                Write-Log "Requirements file not found; skipping dependency install." "WARN"
            }
        } finally {
            Pop-Location
        }
    }
    if (-not (Test-Path $PythonExe)) {
        throw "Python executable not found at $PythonExe"
    }
    Write-Log "Python version: $(& $PythonExe --version)" "INFO"
    if (-not (Test-Path $script:PrefectExe)) {
        throw "Prefect CLI not found at $script:PrefectExe. Install Prefect inside the virtual environment."
    }
    Write-Log "Prefect version: $(& $script:PrefectExe version)" "INFO"
}

function Start-PrefectServer {
    $healthUrl = "$PREFECT_API_URL/health"
    if (Wait-Http-200 -Url $healthUrl -TimeoutSeconds 5 -IntervalSeconds 1) {
        Write-Log "Prefect server already healthy at $PREFECT_UI" "SUCCESS"
        return
    }
    Write-Log "Launching Prefect server..." "INFO"
    $procInfo = Start-Logged -Name "prefect-server" -FilePath $script:PrefectExe -ArgumentList @("server", "start")
    Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
    $script:ServerStarted = $true
    if (-not (Wait-Http-200 -Url $healthUrl -TimeoutSeconds 60 -IntervalSeconds 3)) {
        throw "Prefect server failed health check. Inspect logs at $($procInfo.StdoutLog)"
    }
    Write-Log "Prefect server reachable at $PREFECT_UI" "SUCCESS"
}

function Ensure-WorkPool {
    Write-Log "Ensuring work pool '$WorkPool' exists..." "INFO"
    & $script:PrefectExe work-pool inspect $WorkPool *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Creating work pool '$WorkPool'..." "INFO"
        & $script:PrefectExe work-pool create $WorkPool --type process
        if ($LASTEXITCODE -ne 0) { throw "Failed to create work pool '$WorkPool' (exit $LASTEXITCODE)" }
    } else {
        Write-Log "Work pool '$WorkPool' already present." "SUCCESS"
    }
}

function Deploy-PrefectFlows {
    $spec = Join-Path $PSScriptRoot "prefect.yaml"
    if (-not (Test-Path $spec)) {
        Write-Log "prefect.yaml not found; skipping deployment step." "WARN"
        return
    }
    Write-Log "Deploying Prefect flows from prefect.yaml..." "INFO"
    $previousPromptSetting = $env:PREFECT_CLI_PROMPTS_ENABLED
    $previousDefaultResponse = $env:PREFECT_DEFAULT_RESPONSE
    $env:PREFECT_CLI_PROMPTS_ENABLED = "false"
    $env:PREFECT_DEFAULT_RESPONSE = "n"
    try {
        $args = @("deploy", "--prefect-file", $spec, "--all")
        if ($WorkPool) {
            $args += @("--pool", $WorkPool)
        }
        $responses = @("n","n","n","n","n","n","n","n","n","n")
        $result = Invoke-ProcessWithInput -FilePath $script:PrefectExe -ArgumentList $args -InputLines $responses -Description "prefect deploy"
        if ($result.ExitCode -ne 0) {
            Write-Log "Prefect deployment failed (exit $($result.ExitCode))." "WARN"
            if ($result.Stderr) {
                Write-Log $result.Stderr.Trim() "DEBUG"
            }
            Write-Log "You can retry manually with `prefect deploy --prefect-file prefect.yaml --all --pool $WorkPool` once the server is reachable." "WARN"
            return
        }
        if ($result.Stdout) {
            foreach ($line in ($result.Stdout -split "`r?`n")) {
                if ($line.Trim()) { Write-Log $line.Trim() "DEBUG" }
            }
        }
        Write-Log "Deployments up to date." "SUCCESS"
    }
    finally {
        if ($null -ne $previousPromptSetting) {
            $env:PREFECT_CLI_PROMPTS_ENABLED = $previousPromptSetting
        }
        else {
            Remove-Item Env:PREFECT_CLI_PROMPTS_ENABLED -ErrorAction SilentlyContinue
        }
        if ($null -ne $previousDefaultResponse) {
            $env:PREFECT_DEFAULT_RESPONSE = $previousDefaultResponse
        }
        else {
            Remove-Item Env:PREFECT_DEFAULT_RESPONSE -ErrorAction SilentlyContinue
        }
    }
}

function Start-PrefectWorkerOrAgent {
    if ($UseDockerAgent) {
        Ensure-Command -Command "docker" -FriendlyName "Docker CLI"
        $imageName = "bitmex-prefect-agent"
        & docker image inspect $imageName *> $null
        if ($LASTEXITCODE -ne 0) {
            $dockerfile = Join-Path $PSScriptRoot "docker\prefect-agent.Dockerfile"
            if (-not (Test-Path $dockerfile)) {
                throw "Dockerfile not found at $dockerfile"
            }
            Write-Log "Building Docker image $imageName..." "INFO"
            Push-Location $PSScriptRoot
            try {
                & docker build -f $dockerfile -t $imageName .
                if ($LASTEXITCODE -ne 0) { throw "Docker build failed (exit $LASTEXITCODE)" }
            } finally {
                Pop-Location
            }
        }
        $args = @(
            "run", "--rm", "--network", "host",
            "-e", "PREFECT_API_URL=$PREFECT_API_URL",
            $imageName, "--pool", $WorkPool
        )
        $procInfo = Start-Logged -Name "prefect-agent" -FilePath "docker" -ArgumentList $args
        Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
    } else {
        $procInfo = Start-Logged -Name "prefect-worker" -FilePath $script:PrefectExe -ArgumentList @("worker", "start", "--pool", $WorkPool)
        Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
    }
}

function Start-Gui {
    function Launch-GuiBrowser {
        param(
            [Parameter(Mandatory = $true)][string]$Url
        )
        try {
            Start-Process $Url | Out-Null
            Write-Log "Opening browser at $Url" "INFO"
        } catch {
            Write-Log ("Unable to launch browser for {0}: {1}" -f $Url, $_.Exception.Message) "WARN"
        }
    }

    switch ($GuiMode.ToLowerInvariant()) {
        "flask" {
            $entry = Join-Path $PSScriptRoot $FlaskEntry
            if (-not (Test-Path $entry)) { throw "Flask entry not found at $entry" }
            if (Test-Port -TargetHost "127.0.0.1" -Port 5000) {
                Write-Log "Port 5000 already in use; Flask UI may fail to bind." "WARN"
            }
            try {
                Ensure-UiVendorAssets
            } catch {
                throw "Unable to prepare UI vendor assets: $($_.Exception.Message)"
            }
            try {
                $sourcePath = Join-Path $PSScriptRoot "tradingbotui\static\js\app.jsx"
                $bundlePath = Join-Path $PSScriptRoot "tradingbotui\static\js\app.bundle.js"
                Invoke-FrontendBundle -SourcePath $sourcePath -BundlePath $bundlePath
            } catch {
                Write-Log ("Frontend bundle build failed: {0}" -f $_.Exception.Message) "ERROR"
            }
            $previousFlaskDebug = Get-Item Env:FLASK_DEBUG -ErrorAction SilentlyContinue
            try {
                if ($DebugMode) {
                    $env:FLASK_DEBUG = "1"
                } else {
                    Remove-Item Env:FLASK_DEBUG -ErrorAction SilentlyContinue
                }
                $procInfo = Start-Logged -Name "gui-flask" -FilePath $PythonExe -ArgumentList @("-u", $entry)
            } finally {
                if ($previousFlaskDebug) {
                    $env:FLASK_DEBUG = $previousFlaskDebug.Value
                } else {
                    Remove-Item Env:FLASK_DEBUG -ErrorAction SilentlyContinue
                }
            }
            Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
            Write-Log "Flask UI available at http://127.0.0.1:5000" "SUCCESS"
            Launch-GuiBrowser -Url "http://127.0.0.1:5000"
            if ($procInfo.StdoutLog) {
                Start-LogMonitor -Name "gui-flask-out" -Path $procInfo.StdoutLog
                if (-not $DebugMode) {
                    Start-LogTail -Name "gui-flask-out" -Path $procInfo.StdoutLog
                }
            }
            if ($procInfo.StderrLog) {
                Start-LogMonitor -Name "gui-flask-err" -Path $procInfo.StderrLog
                if (-not $DebugMode) {
                    Start-LogTail -Name "gui-flask-err" -Path $procInfo.StderrLog
                }
            }
        }
        "streamlit" {
            $entry = Join-Path $PSScriptRoot $StreamlitEntry
            if (-not (Test-Path $entry)) { throw "Streamlit entry not found at $entry" }
            if (Test-Port -TargetHost "127.0.0.1" -Port 8501) {
                Write-Log "Port 8501 already in use; Streamlit may fail to bind." "WARN"
            }
            $procInfo = Start-Logged -Name "gui-streamlit" -FilePath $PythonExe -ArgumentList @("-m", "streamlit", "run", $entry, "--server.port", "8501", "--server.headless", "true")
            Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
            Write-Log "Streamlit UI available at http://127.0.0.1:8501" "SUCCESS"
            Launch-GuiBrowser -Url "http://127.0.0.1:8501"
        }
        "fastapi" {
            if (Test-Port -TargetHost "127.0.0.1" -Port 8000) {
                Write-Log "Port 8000 already in use; FastAPI may fail to bind." "WARN"
            }
            $procInfo = Start-Logged -Name "gui-fastapi" -FilePath $PythonExe -ArgumentList @("-m", "uvicorn", $FastAPIApp, "--host", "127.0.0.1", "--port", "8000", "--reload")
            Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
            Write-Log "FastAPI UI available at http://127.0.0.1:8000" "SUCCESS"
            Launch-GuiBrowser -Url "http://127.0.0.1:8000"
        }
        "gradio" {
            $entry = Join-Path $PSScriptRoot $GradioEntry
            if (-not (Test-Path $entry)) { throw "Gradio entry not found at $entry" }
            $procInfo = Start-Logged -Name "gui-gradio" -FilePath $PythonExe -ArgumentList @($entry)
            Register-ManagedProcess -ProcessInfo $procInfo | Out-Null
            Write-Log "Gradio UI starting (check logs for URL)" "SUCCESS"
            Launch-GuiBrowser -Url "http://127.0.0.1:7860"
        }
        default {
            throw "Unsupported GuiMode '$GuiMode'. Use flask, streamlit, fastapi, or gradio."
        }
    }
}

function Start-DebugTails {
    foreach ($entry in $script:ProcessRegistry) {
        if ($entry.StdoutLog) { Start-LogTail -Name "$($entry.Name)-out" -Path $entry.StdoutLog }
        if ($entry.StderrLog) { Start-LogTail -Name "$($entry.Name)-err" -Path $entry.StderrLog }
    }
}

function Wait-ForShutdown {
    Write-Log "RunBot started. Press Ctrl+C to stop, or run RunBot.ps1 -Stop in another shell." "INFO"
    while (-not $script:ShutdownRequested) {
        foreach ($entry in $script:ProcessRegistry) {
            if ($entry.StartedByScript -and $entry.Process) {
                if (-not (Ensure-Process -Process $entry.Process -Name $entry.Name)) {
                    $script:ShutdownRequested = $true
                    break
                }
            }
        }
        Start-Sleep -Seconds 3
    }
}

# -----------------------------------------------------------------------------
# Stop handling (standalone invocation)
# -----------------------------------------------------------------------------
if ($Stop) {
    Stop-RunBotFromState
    return
}

# -----------------------------------------------------------------------------
# Start path
# -----------------------------------------------------------------------------
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

if (Test-Path $StateFile) {
    Write-Log "Existing RunBot state detected; attempting automatic stop before restart." "WARN"
    try {
        Stop-RunBotFromState
        Start-Sleep -Seconds 2
    } catch {
        Write-Log "Automatic stop failed: $($_.Exception.Message)" "ERROR"
        throw "Aborting to avoid duplicate launches."
    }
    if (Test-Path $StateFile) {
        Write-Log "State file still present after stop attempt; aborting launch to avoid duplicates." "ERROR"
        throw "Aborting to avoid duplicate launches."
    }
}

Load-DotEnv -Path $EnvFile

if ($env:RUNBOT_SKIP_KAFKA) {
    $flag = $env:RUNBOT_SKIP_KAFKA.ToLowerInvariant()
    if ($flag -in @("1", "true", "yes")) {
        $ManageKafkaStack = $false
        Write-Log "RUNBOT_SKIP_KAFKA set – Kafka/ZooKeeper stack management disabled." "WARN"
    }
}

$env:PREFECT_API_URL = $PREFECT_API_URL
if ($DebugMode) {
    $env:PYTHONUNBUFFERED = "1"
    $env:PREFECT_LOGGING_LEVEL = "DEBUG"
}

try {
    Start-KafkaStack
    Initialize-Python
    Start-PrefectServer
    Ensure-WorkPool
    Deploy-PrefectFlows
    Start-PrefectWorkerOrAgent
    Start-Gui
    Write-Log "Smoke test helpers:" "INFO"
    Write-Log "  prefect deployment run 'bitmex-qc-flow/qc-bitmex'" "INFO"
    Write-Log "  prefect deployment run 'bitmex-daily-flow/daily-bitmex'" "INFO"
    if (Test-Path (Join-Path $PSScriptRoot "etl-params.json")) {
        Write-Log "  prefect deployment run 'bitmex-etl-flow/etl-bitmex' --params-file etl-params.json" "INFO"
    } else {
        Write-Log "  prefect deployment run 'bitmex-etl-flow/etl-bitmex'" "INFO"
    }
    Save-RunState
    if ($DebugMode) {
        Start-DebugTails
    }
    Wait-ForShutdown
} catch {
    if ($_.Exception -is [System.Management.Automation.Host.HostException] -or $_.Exception.Message -match "The pipeline has been stopped") {
        $script:ShutdownRequested = $true
        Write-Log "Shutdown requested. Cleaning up..." "WARN"
    } else {
        Write-Log $_.Exception.Message "ERROR"
        throw
    }
} finally {
    Stop-RunBotCurrent
}

# -----------------------------------------------------------------------------
# Packaging Notes
# -----------------------------------------------------------------------------
# To build RunBot.exe:
#   Install-Module ps2exe
#   Invoke-ps2exe .\RunBot.ps1 .\RunBot.exe
