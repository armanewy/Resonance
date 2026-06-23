param(
    [switch]$AllowEarly,
    [int]$Hours = 720,
    [int]$EarlyHours = 24,
    [int]$MaxLagSeconds = 3600,
    [int]$EarlyMaxLagSeconds = 0,
    [int]$AuditHours = 24,
    [int]$ScanHours = 168,
    [string]$ExpectedBranch = "product/resonance-app",
    [string]$RequiredAncestorCommit = "f7e3b34b2d54dcaf3cecbe0e59517bea36a090aa"
)

$ErrorActionPreference = "Stop"

function New-StepName {
    param([int]$Number, [string]$Name, [string]$Extension = "txt")
    return ("{0:D2}_{1}.{2}" -f $Number, $Name, $Extension)
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$OutFile
    )
    $output = & $FilePath @Arguments 2>&1
    $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    $text = ($output | Out-String)
    Set-Content -LiteralPath $OutFile -Value $text -Encoding UTF8
    return [pscustomobject]@{
        exit_code = $exitCode
        output = $text
        path = $OutFile
    }
}

function ConvertFrom-JsonOutput {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $null
    }
    return ($Text | ConvertFrom-Json)
}

function Write-OperatorResult {
    param([hashtable]$Result, [string]$Path)
    $Result.updated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $Result | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Add-CommandRecord {
    param([System.Collections.IDictionary]$Result, [string]$Name, [object]$CommandResult)
    $records = @($Result["commands"])
    $records += [ordered]@{
        name = $Name
        exit_code = $CommandResult.exit_code
        output_path = $CommandResult.path
    }
    $Result["commands"] = $records
}

function Test-GitAncestor {
    param([string]$AncestorCommit)
    & git merge-base --is-ancestor $AncestorCommit HEAD *> $null
    return ($LASTEXITCODE -eq 0)
}

function New-LinearHypothesis {
    param(
        [string]$Title,
        [string]$Claim,
        [string]$Rationale,
        [string]$Target,
        [string]$InputMetric,
        [string]$NegativeControl,
        [int]$LagSeconds,
        [string]$TargetTransform,
        [string]$CatalogId,
        [int]$Seed
    )
    return [ordered]@{
        schema_version = "1.0"
        hypothesis_type = "observational_prediction"
        title = $Title
        concise_claim = $Claim
        rationale = $Rationale
        target_metric = $Target
        input_metrics = @($InputMetric)
        target_transform = $TargetTransform
        expression = [ordered]@{
            node = "add"
            left = [ordered]@{
                node = "multiply"
                left = [ordered]@{ node = "fitted_parameter"; parameter = "scale" }
                right = [ordered]@{
                    node = "lag"
                    input = [ordered]@{ node = "metric"; metric = $InputMetric }
                    lag_seconds = $LagSeconds
                }
            }
            right = [ordered]@{ node = "fitted_parameter"; parameter = "offset" }
        }
        parameter_bounds = [ordered]@{
            scale = [ordered]@{ lower = -1000.0; upper = 1000.0 }
            offset = [ordered]@{ lower = -1000.0; upper = 1000.0 }
        }
        expected_direction = "positive"
        maximum_lag_seconds = $LagSeconds
        fitting_metric = "rmse"
        tuning_metric = "rmse"
        blind_metrics = @("mae", "rmse", "spearman_r")
        minimum_blind_effect = 0.1
        minimum_baseline_improvement = 0.05
        negative_controls = @(
            [ordered]@{
                metric = $NegativeControl
                rationale = "A distinct local signal should not explain the same transformed latency relationship."
            }
        )
        falsification_conditions = @(
            [ordered]@{ description = "Tuning performance does not improve over the preregistered baseline." },
            [ordered]@{ description = "The negative control is associated with the prediction." }
        )
        complexity_budget = [ordered]@{ max_ast_nodes = 8; max_source_metrics = 1 }
        origin = "manual"
        parent_hypothesis_ids = @()
        snapshot_metric_catalog_id = $CatalogId
        random_seed = $Seed
    }
}

function New-InteractionHypothesis {
    param(
        [string]$Target,
        [string]$FirstMetric,
        [string]$SecondMetric,
        [string]$NegativeControl,
        [int]$LagSeconds,
        [string]$TargetTransform,
        [string]$CatalogId
    )
    return [ordered]@{
        schema_version = "1.0"
        hypothesis_type = "observational_prediction"
        title = "Upload and CPU interaction predicts transformed TCP latency"
        concise_claim = "Recent upload throughput and CPU activity together are associated with transformed TCP latency."
        rationale = "Network contention may matter most when local CPU activity is also elevated; this checks a simple bounded interaction."
        target_metric = $Target
        input_metrics = @($FirstMetric, $SecondMetric)
        target_transform = $TargetTransform
        expression = [ordered]@{
            node = "add"
            left = [ordered]@{
                node = "multiply"
                left = [ordered]@{ node = "fitted_parameter"; parameter = "scale" }
                right = [ordered]@{
                    node = "multiply"
                    left = [ordered]@{
                        node = "lag"
                        input = [ordered]@{ node = "metric"; metric = $FirstMetric }
                        lag_seconds = $LagSeconds
                    }
                    right = [ordered]@{
                        node = "lag"
                        input = [ordered]@{ node = "metric"; metric = $SecondMetric }
                        lag_seconds = $LagSeconds
                    }
                }
            }
            right = [ordered]@{ node = "fitted_parameter"; parameter = "offset" }
        }
        parameter_bounds = [ordered]@{
            scale = [ordered]@{ lower = -1.0; upper = 1.0 }
            offset = [ordered]@{ lower = -1000.0; upper = 1000.0 }
        }
        expected_direction = "positive"
        maximum_lag_seconds = $LagSeconds
        fitting_metric = "rmse"
        tuning_metric = "rmse"
        blind_metrics = @("mae", "rmse", "spearman_r")
        minimum_blind_effect = 0.1
        minimum_baseline_improvement = 0.05
        negative_controls = @(
            [ordered]@{
                metric = $NegativeControl
                rationale = "A distinct local signal should not explain the same transformed latency relationship."
            }
        )
        falsification_conditions = @(
            [ordered]@{ description = "Tuning performance does not improve over the preregistered baseline." },
            [ordered]@{ description = "The negative control is associated with the prediction." }
        )
        complexity_budget = [ordered]@{ max_ast_nodes = 12; max_source_metrics = 2 }
        origin = "manual"
        parent_hypothesis_ids = @()
        snapshot_metric_catalog_id = $CatalogId
        random_seed = 8675309
    }
}

function New-ProposalFile {
    param(
        [object]$Manifest,
        [string[]]$AvailableMetrics,
        [string]$Path,
        [int]$LagSeconds,
        [string]$TargetTransform
    )
    $target = "tcp_latency_ms"
    $candidateInputs = @(
        "network_sent_bytes_per_second",
        "cpu_percent",
        "battery_plugged",
        "battery_percent"
    ) | Where-Object { $AvailableMetrics -contains $_ }
    $candidateControls = @(
        "memory_percent",
        "network_recv_bytes_per_second",
        "dns_latency_ms",
        "battery_percent",
        "battery_plugged",
        "cpu_percent"
    ) | Where-Object { $AvailableMetrics -contains $_ -and $_ -ne $target }

    $proposals = @()
    $catalogId = [string]$Manifest.metric_catalog.catalog_id
    $seed = 8675309
    foreach ($inputMetric in $candidateInputs) {
        $control = $candidateControls | Where-Object { $_ -ne $inputMetric } | Select-Object -First 1
        if (-not $control) {
            continue
        }
        $label = switch ($inputMetric) {
            "network_sent_bytes_per_second" { "Upload throughput" }
            "cpu_percent" { "CPU activity" }
            "battery_plugged" { "Charging state" }
            "battery_percent" { "Battery level" }
            default { $inputMetric }
        }
        $proposals += New-LinearHypothesis `
            -Title "$label predicts transformed TCP latency" `
            -Claim "$label is associated with transformed TCP latency." `
            -Rationale "This proposal exercises the sealed latency question using only already collected local signals." `
            -Target $target `
            -InputMetric $inputMetric `
            -NegativeControl $control `
            -LagSeconds $LagSeconds `
            -TargetTransform $TargetTransform `
            -CatalogId $catalogId `
            -Seed $seed
        $seed += 1
    }

    if (($AvailableMetrics -contains "network_sent_bytes_per_second") -and ($AvailableMetrics -contains "cpu_percent")) {
        $control = $candidateControls | Where-Object { $_ -notin @("network_sent_bytes_per_second", "cpu_percent") } | Select-Object -First 1
        if ($control) {
            $proposals += New-InteractionHypothesis `
                -Target $target `
                -FirstMetric "network_sent_bytes_per_second" `
                -SecondMetric "cpu_percent" `
                -NegativeControl $control `
                -LagSeconds $LagSeconds `
                -TargetTransform $TargetTransform `
                -CatalogId $catalogId
        }
    }

    [ordered]@{ proposals = $proposals } | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath $Path -Encoding UTF8
    return $proposals.Count
}

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $repo "data\science\operator_logs\latency_$timestamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$resultPath = Join-Path $runDir "operator_result.json"

$branch = (git branch --show-current).Trim()
$commit = (git rev-parse HEAD).Trim()
$result = [ordered]@{
    status = "STARTED"
    objective = "sealed_latency_prediction_non_blind_rehearsal"
    allow_early = [bool]$AllowEarly
    started_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    updated_at_utc = $null
    run_dir = $runDir
    branch = $branch
    commit = $commit
    expected_branch = $ExpectedBranch
    required_ancestor_commit = $RequiredAncestorCommit
    blind_budget_spent = $false
    commands = @()
    warnings = @()
}
Write-OperatorResult -Result $result -Path $resultPath

if ($branch -ne $ExpectedBranch) {
    $result.status = "WRONG_BRANCH"
    $result.error = "Expected branch $ExpectedBranch but found $branch."
    Write-OperatorResult -Result $result -Path $resultPath
    throw $result.error
}
if (-not (Test-GitAncestor -AncestorCommit $RequiredAncestorCommit)) {
    $result.status = "WRONG_LINEAGE"
    $result.error = "Current HEAD $commit does not descend from required ancestor $RequiredAncestorCommit."
    Write-OperatorResult -Result $result -Path $resultPath
    throw $result.error
}
$result.lineage_verified = $true
Write-OperatorResult -Result $result -Path $resultPath

$step = 1
$auditOut = Join-Path $runDir (New-StepName $step "audit" "json")
$audit = Invoke-Captured -FilePath $python -Arguments @("-m", "resonance.audit", "--hours", "$AuditHours", "--json") -OutFile $auditOut
Add-CommandRecord -Result $result -Name "audit" -CommandResult $audit
if ($audit.exit_code -ne 0) {
    $result.status = "AUDIT_FAILED"
    $result.error = $audit.output
    Write-OperatorResult -Result $result -Path $resultPath
    throw "Audit failed; see $auditOut"
}
$auditJson = ConvertFrom-JsonOutput $audit.output
$result.audit = [ordered]@{
    hours = $AuditHours
    total_measurements = $auditJson.total_measurements
    total_collector_errors = $auditJson.total_collector_errors
    stale_metrics = @($auditJson.stale_metrics)
    low_coverage_metrics = @($auditJson.metrics_with_less_than_80_percent_coverage)
}
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$ledgerOut = Join-Path $runDir (New-StepName $step "ledger_verify" "txt")
$ledger = Invoke-Captured -FilePath $python -Arguments @("-m", "resonance.science.ledger_cli", "verify") -OutFile $ledgerOut
Add-CommandRecord -Result $result -Name "ledger_verify" -CommandResult $ledger
if ($ledger.exit_code -ne 0) {
    $result.status = "LEDGER_VERIFY_FAILED"
    $result.error = $ledger.output
    Write-OperatorResult -Result $result -Path $resultPath
    throw "Ledger verification failed; see $ledgerOut"
}
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$scanOut = Join-Path $runDir (New-StepName $step "scan_dry_run" "json")
$scan = Invoke-Captured -FilePath $python -Arguments @("-m", "resonance.scan", "--hours", "$ScanHours", "--dry-run", "--json") -OutFile $scanOut
Add-CommandRecord -Result $result -Name "scan_dry_run" -CommandResult $scan
if ($scan.exit_code -ne 0) {
    $result.warnings += "Scanner dry-run failed; continuing to sealed-loop rehearsal."
}
Write-OperatorResult -Result $result -Path $resultPath

$availableMetrics = @(
    $auditJson.metrics |
        Where-Object { $_.sample_count -gt 0 } |
        ForEach-Object { [string]$_.metric }
)
$result.available_metrics = $availableMetrics
if ($availableMetrics -notcontains "tcp_latency_ms") {
    $result.status = "NOT_READY_NO_TCP_LATENCY"
    $result.error = "tcp_latency_ms is not present in the current audit window."
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}

$snapshotMetrics = @(
    "tcp_latency_ms",
    "network_sent_bytes_per_second",
    "cpu_percent",
    "battery_plugged",
    "battery_percent",
    "memory_percent",
    "network_recv_bytes_per_second",
    "dns_latency_ms"
) | Where-Object { $availableMetrics -contains $_ } | Select-Object -Unique

if ($snapshotMetrics.Count -lt 3) {
    $result.status = "NOT_READY_TOO_FEW_METRICS"
    $result.error = "Need tcp_latency_ms plus at least one input and one negative-control metric."
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}

$snapshotHours = if ($AllowEarly) { $EarlyHours } else { $Hours }
$snapshotMaxLagSeconds = if ($AllowEarly) { $EarlyMaxLagSeconds } else { $MaxLagSeconds }
$metricCsv = ($snapshotMetrics -join ",")
$result.snapshot_request = [ordered]@{
    hours = $snapshotHours
    max_lag_seconds = $snapshotMaxLagSeconds
    metrics = $snapshotMetrics
}
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$snapshotOut = Join-Path $runDir (New-StepName $step "snapshot_create" "json")
$snapshot = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "snapshot", "create",
    "--db", "data/resonance.db",
    "--hours", "$snapshotHours",
    "--metrics", $metricCsv,
    "--max-lag-seconds", "$snapshotMaxLagSeconds"
) -OutFile $snapshotOut
Add-CommandRecord -Result $result -Name "snapshot_create" -CommandResult $snapshot
if ($snapshot.exit_code -ne 0) {
    $result.status = "SNAPSHOT_NOT_READY"
    $result.error = $snapshot.output
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}
$snapshotJson = ConvertFrom-JsonOutput $snapshot.output
$snapshotId = [string]$snapshotJson.snapshot_id
$result.snapshot_id = $snapshotId
$result.snapshot_row_counts = $snapshotJson.manifest.row_counts
$result.snapshot_coverage = $snapshotJson.manifest.coverage
Write-OperatorResult -Result $result -Path $resultPath

$proposalPath = Join-Path $runDir "latency_hypotheses.json"
$hypothesisLagSeconds = [Math]::Min($snapshotMaxLagSeconds, 300)
$targetTransform = if ($AllowEarly) { "identity" } else { "robust_zscore" }
$proposalCount = New-ProposalFile `
    -Manifest $snapshotJson.manifest `
    -AvailableMetrics $snapshotMetrics `
    -Path $proposalPath `
    -LagSeconds $hypothesisLagSeconds `
    -TargetTransform $targetTransform
$result.proposal_file = $proposalPath
$result.proposal_count = $proposalCount
$result.target_transform = $targetTransform
if ($proposalCount -le 0) {
    $result.status = "NO_VALID_PROPOSALS"
    $result.error = "No latency hypotheses could be generated from the available local metric set."
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$imagineOut = Join-Path $runDir (New-StepName $step "imagine" "json")
$imagine = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "imagine",
    "--snapshot", $snapshotId,
    "--provider", "file",
    "--provider-file", $proposalPath,
    "--max-hypotheses", "8"
) -OutFile $imagineOut
Add-CommandRecord -Result $result -Name "imagine" -CommandResult $imagine
if ($imagine.exit_code -ne 0) {
    $result.status = "IMAGINE_FAILED"
    $result.error = $imagine.output
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}
$imagineJson = ConvertFrom-JsonOutput $imagine.output
$runId = [string]$imagineJson.run_id
$result.imagination_run_id = $runId
$result.accepted_review_count = $imagineJson.accepted_review_count
$result.rejected_provider_count = $imagineJson.rejected_provider_count
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$reviewOut = Join-Path $runDir (New-StepName $step "review" "json")
$review = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "review",
    $runId
) -OutFile $reviewOut
Add-CommandRecord -Result $result -Name "review" -CommandResult $review
if ($review.exit_code -ne 0) {
    $result.status = "REVIEW_FAILED"
    $result.error = $review.output
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}
$reviewJson = ConvertFrom-JsonOutput $review.output
$acceptedIndexes = @(
    $reviewJson.proposals |
        Where-Object { $_.status -eq "review_accepted" } |
        ForEach-Object { [int]$_.index }
)
$result.review_accepted_indexes = $acceptedIndexes
if ($acceptedIndexes.Count -eq 0) {
    $result.status = "NO_VALID_PROPOSALS"
    $result.error = "Deterministic review rejected every generated latency hypothesis."
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}
Write-OperatorResult -Result $result -Path $resultPath

foreach ($index in $acceptedIndexes) {
    $step += 1
    $approveOut = Join-Path $runDir (New-StepName $step "approve_$index" "json")
    $approve = Invoke-Captured -FilePath $python -Arguments @(
        "-m", "resonance.science.cli",
        "review",
        $runId,
        "--approve", "$index"
    ) -OutFile $approveOut
    Add-CommandRecord -Result $result -Name "approve_$index" -CommandResult $approve
    if ($approve.exit_code -ne 0) {
        $result.status = "APPROVAL_FAILED"
        $result.error = $approve.output
        Write-OperatorResult -Result $result -Path $resultPath
        exit 0
    }
}
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$fitOut = Join-Path $runDir (New-StepName $step "fit_approved" "json")
$fit = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "fit-approved",
    $runId
) -OutFile $fitOut
Add-CommandRecord -Result $result -Name "fit_approved" -CommandResult $fit
if ($fit.exit_code -ne 0) {
    $result.status = "FIT_OR_TUNING_FAILED"
    $result.error = $fit.output
    Write-OperatorResult -Result $result -Path $resultPath
    exit 0
}
$fitJson = ConvertFrom-JsonOutput $fit.output
$result.fit_result_count = @($fitJson.fit_results).Count
$result.tuning = $fitJson.tuning
$result.selected_candidate_id = $fitJson.selected_candidate_id
if ([string]::IsNullOrWhiteSpace([string]$fitJson.selected_candidate_id)) {
    $result.status = "NO_TUNING_WINNER"
} else {
    $result.status = "TUNING_WINNER_SELECTED"
}
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$finalLedgerOut = Join-Path $runDir (New-StepName $step "ledger_verify_final" "txt")
$finalLedger = Invoke-Captured -FilePath $python -Arguments @("-m", "resonance.science.ledger_cli", "verify") -OutFile $finalLedgerOut
Add-CommandRecord -Result $result -Name "ledger_verify_final" -CommandResult $finalLedger
if ($finalLedger.exit_code -ne 0) {
    $result.status = "LEDGER_VERIFY_FAILED_AFTER_REHEARSAL"
    $result.error = $finalLedger.output
}
Write-OperatorResult -Result $result -Path $resultPath

Write-Output "Latency experiment dry rehearsal complete."
Write-Output "Run directory: $runDir"
Write-Output "Status: $($result.status)"
Write-Output "Operator result: $resultPath"
