param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [switch]$ConfirmBlindSpend,
    [string]$ExpectedBranch = "product/resonance-app",
    [string]$RequiredAncestorCommit = "f7e3b34b2d54dcaf3cecbe0e59517bea36a090aa"
)

$ErrorActionPreference = "Stop"

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
    param([object]$Result, [string]$Path)
    $Result.updated_at_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $Result | ConvertTo-Json -Depth 80 | Set-Content -LiteralPath $Path -Encoding UTF8
}

function Add-CommandRecord {
    param([object]$Result, [string]$Name, [object]$CommandResult)
    if ($null -eq $Result.commands) {
        $Result | Add-Member -NotePropertyName commands -NotePropertyValue @()
    }
    $records = @($Result.commands)
    $records += [ordered]@{
        name = $Name
        exit_code = $CommandResult.exit_code
        output_path = $CommandResult.path
    }
    $Result.commands = $records
}

function Test-GitAncestor {
    param([string]$AncestorCommit)
    & git merge-base --is-ancestor $AncestorCommit HEAD *> $null
    return ($LASTEXITCODE -eq 0)
}

if (-not $ConfirmBlindSpend) {
    throw "Refusing to spend blind budget without -ConfirmBlindSpend."
}

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$resolvedRunDir = (Resolve-Path -LiteralPath $RunDir).Path
$resultPath = Join-Path $resolvedRunDir "operator_result.json"
if (-not (Test-Path $resultPath)) {
    throw "operator_result.json not found in $resolvedRunDir"
}

$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
$branch = (git branch --show-current).Trim()
$commit = (git rev-parse HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Expected branch $ExpectedBranch but found $branch."
}
if (-not (Test-GitAncestor -AncestorCommit $RequiredAncestorCommit)) {
    throw "Current HEAD $commit does not descend from required ancestor $RequiredAncestorCommit."
}
if ([string]::IsNullOrWhiteSpace([string]$result.commit)) {
    throw "operator_result.json does not record the driver commit."
}
if ($commit -ne [string]$result.commit) {
    throw "Current commit $commit does not match the run commit $($result.commit). Re-check out the exact commit that selected the tuning winner."
}
if ($result.status -ne "TUNING_WINNER_SELECTED") {
    throw "Run status must be TUNING_WINNER_SELECTED before blind spend; found $($result.status)."
}
if ([string]::IsNullOrWhiteSpace([string]$result.selected_candidate_id)) {
    throw "operator_result.json does not contain selected_candidate_id."
}
if ($result.blind_budget_spent -eq $true) {
    throw "operator_result.json already records blind_budget_spent=true."
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$step = 1
$preregOut = Join-Path $resolvedRunDir ("blind_{0:D2}_preregister.json" -f $step)
$prereg = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "preregister",
    "--candidate", ([string]$result.selected_candidate_id)
) -OutFile $preregOut
Add-CommandRecord -Result $result -Name "blind_preregister" -CommandResult $prereg
if ($prereg.exit_code -ne 0) {
    $result.status = "BLIND_PREREGISTRATION_FAILED"
    $result.error = $prereg.output
    Write-OperatorResult -Result $result -Path $resultPath
    throw "Preregistration failed; see $preregOut"
}
$preregJson = ConvertFrom-JsonOutput $prereg.output
$result.preregistration_id = $preregJson.preregistration_id
$result.blind_budget_spent = $true
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$blindOut = Join-Path $resolvedRunDir ("blind_{0:D2}_blind_evaluate.json" -f $step)
$blind = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "blind-evaluate",
    ([string]$result.preregistration_id)
) -OutFile $blindOut
Add-CommandRecord -Result $result -Name "blind_evaluate" -CommandResult $blind
if ($blind.exit_code -ne 0) {
    $result.status = "BLIND_EVALUATION_FAILED_AFTER_BUDGET_SPEND"
    $result.error = $blind.output
    Write-OperatorResult -Result $result -Path $resultPath
    throw "Blind evaluation failed after budget spend; see $blindOut"
}
$blindJson = ConvertFrom-JsonOutput $blind.output
$result.blind_evaluation = $blindJson
Write-OperatorResult -Result $result -Path $resultPath

$step += 1
$reportOut = Join-Path $resolvedRunDir ("blind_{0:D2}_report.json" -f $step)
$report = Invoke-Captured -FilePath $python -Arguments @(
    "-m", "resonance.science.cli",
    "report",
    ([string]$result.preregistration_id)
) -OutFile $reportOut
Add-CommandRecord -Result $result -Name "blind_report" -CommandResult $report
if ($report.exit_code -ne 0) {
    $result.status = "BLIND_REPORT_FAILED"
    $result.error = $report.output
    Write-OperatorResult -Result $result -Path $resultPath
    throw "Blind report failed; see $reportOut"
}
$reportJson = ConvertFrom-JsonOutput $report.output
$result.blind_report = $reportJson

$step += 1
$ledgerOut = Join-Path $resolvedRunDir ("blind_{0:D2}_ledger_verify.txt" -f $step)
$ledger = Invoke-Captured -FilePath $python -Arguments @("-m", "resonance.science.ledger_cli", "verify") -OutFile $ledgerOut
Add-CommandRecord -Result $result -Name "blind_ledger_verify" -CommandResult $ledger
if ($ledger.exit_code -ne 0) {
    $result.status = "LEDGER_VERIFY_FAILED_AFTER_BLIND"
    $result.error = $ledger.output
    Write-OperatorResult -Result $result -Path $resultPath
    throw "Ledger verification failed after blind; see $ledgerOut"
}

$result.status = "BLIND_EVALUATION_COMPLETE"
Write-OperatorResult -Result $result -Path $resultPath

Write-Output "Blind evaluation complete."
Write-Output "Run directory: $resolvedRunDir"
Write-Output "Preregistration: $($result.preregistration_id)"
Write-Output "Operator result: $resultPath"
