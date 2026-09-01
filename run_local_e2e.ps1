[CmdletBinding()]
param(
    [ValidateRange(30, 1800)]
    [int]$StartupTimeoutSeconds = 300,

    [ValidateRange(30, 1800)]
    [int]$E2eTimeoutSeconds = 180,

    # Use this to retain the locally started gateway and worker for manual testing.
    [switch]$KeepLocalProcesses,

    # Compose services are left running by default so their state is available after the test.
    [switch]$TearDownCompose,

    [string]$MockModel = "hf-internal-testing/tiny-random-distilbert"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:GatewayProcess = $null
$script:WorkerProcess = $null
$script:OriginalEnvironment = @{}
$script:CancelHandlerRegistered = $false

function Write-Status([string]$Message) {
    Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $Message) -ForegroundColor Cyan
}

function Invoke-Native([string]$Description, [scriptblock]$Command) {
    Write-Status $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function Get-PythonExecutable {
    $venvPython = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"
    if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
        return $venvPython
    }

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $launcher) {
        throw "Python 3 was not found. Install Python 3 (including the 'py' launcher), then rerun this script."
    }

    Invoke-Native "Creating virtual environment at .venv" { & $launcher.Source -3 -m venv (Join-Path $PSScriptRoot ".venv") }
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        throw "Virtual environment creation completed but $venvPython does not exist."
    }
    return $venvPython
}

function Wait-Until([string]$Name, [scriptblock]$Probe, [int]$TimeoutSeconds, [scriptblock]$FailureProbe = {}) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lastError = $null
    while ((Get-Date) -lt $deadline) {
        try {
            if (& $Probe) {
                Write-Status "$Name is ready."
                return
            }
        }
        catch {
            $lastError = $_.Exception.Message
        }
        if (& $FailureProbe) { break }
        Start-Sleep -Seconds 1 # Poll interval; readiness is always determined by the probe above.
    }
    $detail = if ($lastError) { " Last probe error: $lastError" } else { "" }
    throw "Timed out waiting for $Name after $TimeoutSeconds seconds.$detail"
}

function Test-ComposeServiceHealthy([string]$Service) {
    $containerId = (& docker compose -f $script:ComposeFile ps -q $Service).Trim()
    if ([string]::IsNullOrWhiteSpace($containerId)) { return $false }
    $health = (& docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' $containerId).Trim()
    return $health -eq "healthy"
}

function Test-ProcessStopped([System.Diagnostics.Process]$Process, [string]$Name) {
    if ($null -eq $Process) { return $false }
    $Process.Refresh()
    if ($Process.HasExited) {
        throw "$Name exited unexpectedly (exit code $($Process.ExitCode)). Check logs in .local-e2e-logs."
    }
    return $false
}

function Start-LocalProcess([string]$Name, [string]$Python, [string[]]$Arguments, [string]$LogDirectory) {
    $stdout = Join-Path $LogDirectory "$Name.stdout.log"
    $stderr = Join-Path $LogDirectory "$Name.stderr.log"
    Write-Status "Starting $Name. Logs: $stdout, $stderr"
    return Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}

function Stop-LocalProcess([System.Diagnostics.Process]$Process, [string]$Name) {
    if ($null -ne $Process) {
        $Process.Refresh()
        if (-not $Process.HasExited) {
            Write-Status "Stopping $Name (PID $($Process.Id))."
            Stop-Process -Id $Process.Id -ErrorAction Stop
        }
    }
}

function Show-PostTestStatus {
    Write-Host "`nRabbitMQ queue counts (including the DLQ):" -ForegroundColor Yellow
    Invoke-Native "Reading RabbitMQ queue counts" {
        docker compose -f $script:ComposeFile exec -T rabbitmq rabbitmqctl list_queues name messages
    }

    Write-Host "`nPostgreSQL inference task status:" -ForegroundColor Yellow
    Invoke-Native "Reading PostgreSQL task status" {
        docker compose -f $script:ComposeFile exec -T postgres psql -U inference -d inference `
            -c "SELECT status, COUNT(*) AS task_count FROM inference_tasks GROUP BY status ORDER BY status;"
    }
}

$script:ComposeFile = Join-Path $PSScriptRoot "docker-compose.yml"
$logDirectory = Join-Path $PSScriptRoot ".local-e2e-logs"
$script:CancelHandler = [ConsoleCancelEventHandler]{
    param($sender, $eventArgs)
    Write-Warning "Interrupt received; terminating local pipeline processes and running cleanup."
    foreach ($process in @($script:WorkerProcess, $script:GatewayProcess)) {
        if ($null -ne $process) {
            try {
                $process.Refresh()
                if (-not $process.HasExited) { $process.Kill() }
            }
            catch {
                # The finally block will make a second, best-effort cleanup attempt.
            }
        }
    }
    # Let PowerShell stop the active command; its enclosing finally block performs all cleanup.
    $eventArgs.Cancel = $false
}

try {
    [Console]::add_CancelKeyPress($script:CancelHandler)
    $script:CancelHandlerRegistered = $true
    if (-not (Test-Path -LiteralPath $script:ComposeFile -PathType Leaf)) {
        throw "docker-compose.yml was not found at $script:ComposeFile."
    }
    Set-Location -LiteralPath $PSScriptRoot
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

    Invoke-Native "Checking Docker daemon" { docker info --format '{{.ServerVersion}}' | Out-Null }
    Invoke-Native "Checking Docker Compose" { docker compose version | Out-Null }
    Invoke-Native "Starting PostgreSQL and RabbitMQ with Docker Compose" { docker compose -f $script:ComposeFile up -d }
    Wait-Until "PostgreSQL" { Test-ComposeServiceHealthy "postgres" } $StartupTimeoutSeconds
    Wait-Until "RabbitMQ" { Test-ComposeServiceHealthy "rabbitmq" } $StartupTimeoutSeconds

    $python = Get-PythonExecutable
    Invoke-Native "Upgrading pip" { & $python -m pip install --upgrade pip }
    foreach ($requirementsFile in @("requirements.txt", "requirements-worker.txt", "requirements-test.txt")) {
        $requirementsPath = Join-Path $PSScriptRoot $requirementsFile
        if (-not (Test-Path -LiteralPath $requirementsPath -PathType Leaf)) {
            throw "Required dependency file is missing: $requirementsPath"
        }
        Invoke-Native "Installing dependencies from $requirementsFile" { & $python -m pip install -r $requirementsPath }
    }

    $adapterDirectory = Join-Path $PSScriptRoot "outputs\\adapter"
    $adapterConfig = Join-Path $adapterDirectory "adapter_config.json"
    $adapterWeights = Join-Path $adapterDirectory "adapter_model.safetensors"
    if ((Test-Path -LiteralPath $adapterConfig -PathType Leaf) -and (Test-Path -LiteralPath $adapterWeights -PathType Leaf)) {
        Write-Status "Found existing LoRA adapter at $adapterDirectory."
    }
    else {
        foreach ($dataFile in @("train.csv", "val.csv")) {
            if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $dataFile) -PathType Leaf)) {
                throw "Cannot create a mock LoRA adapter: sample data is missing: $dataFile"
            }
        }
        $env:WANDB_MODE = "offline"
        Write-Status "No local LoRA adapter found; running a one-epoch mock training job."
        Invoke-Native "Training mock LoRA adapter" {
            & $python train_lora_classifier.py --train-csv train.csv --val-csv val.csv --output-dir outputs `
                --model-name $MockModel --num-labels 2 --epochs 1 --batch-size 2 --max-length 64 --patience 1 `
                --wandb-project local-e2e --wandb-run-name mock-lora
        }
        if (-not ((Test-Path -LiteralPath $adapterConfig -PathType Leaf) -and (Test-Path -LiteralPath $adapterWeights -PathType Leaf))) {
            throw "Mock training completed without producing a valid LoRA adapter in $adapterDirectory."
        }
    }

    foreach ($name in @("DATABASE_URL", "AMQP_URL", "LOCAL_MODEL_REGISTRY_DIR", "E2E_TIMEOUT_SECONDS")) {
        $script:OriginalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }
    $env:DATABASE_URL = "postgresql://inference:inference_local_password@127.0.0.1:5432/inference"
    $env:AMQP_URL = "amqp://inference:inference_local_password@127.0.0.1:5672/"
    $env:LOCAL_MODEL_REGISTRY_DIR = $adapterDirectory
    $env:E2E_TIMEOUT_SECONDS = $E2eTimeoutSeconds.ToString()

    $script:GatewayProcess = Start-LocalProcess "gateway" $python @("-m", "uvicorn", "inference_gateway:app", "--host", "127.0.0.1", "--port", "8000") $logDirectory
    Wait-Until "inference gateway" {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/metrics" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } $StartupTimeoutSeconds { Test-ProcessStopped $script:GatewayProcess "Gateway" }

    $script:WorkerProcess = Start-LocalProcess "worker" $python @("inference_worker.py") $logDirectory
    Wait-Until "inference worker" {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:8001/metrics" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -eq 200
    } $StartupTimeoutSeconds { Test-ProcessStopped $script:WorkerProcess "Worker" }

    Invoke-Native "Running end-to-end pipeline test" { & $python test_e2e_pipeline.py }
    Show-PostTestStatus
    Write-Host "`nLocal end-to-end pipeline completed successfully." -ForegroundColor Green
}
finally {
    if (-not $KeepLocalProcesses) {
        Stop-LocalProcess $script:WorkerProcess "worker"
        Stop-LocalProcess $script:GatewayProcess "gateway"
    }
    foreach ($name in $script:OriginalEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $script:OriginalEnvironment[$name], "Process")
    }
    if ($TearDownCompose) {
        Write-Status "Stopping Docker Compose services."
        & docker compose -f $script:ComposeFile down
    }
    if ($script:CancelHandlerRegistered) {
        [Console]::remove_CancelKeyPress($script:CancelHandler)
    }
}
