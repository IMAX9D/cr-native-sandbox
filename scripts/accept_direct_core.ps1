param(
    [ValidateRange(1, 100)]
    [int]$Runs = 10,
    [string]$ExpectedHash = "96598dc9028e1802",
    [string]$EvidenceRoot = $(if ($env:CR_SANDBOX_DATA) { Join-Path $env:CR_SANDBOX_DATA "acceptance-direct-core" } else { throw "Missing CR_SANDBOX_DATA; dot-source runtime.env.ps1 first" })
)

# Requires an already-booted, rootable AVD with the matching package installed.
# Use scripts\smoke.ps1 as the unified entry; it boots the AVD, installs the
# package and then runs this acceptance pass.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ExpectedTowerHp = @(4824, 3052, 3052, 4824, 3052, 3052)

& (Join-Path $PSScriptRoot "build_probe.ps1") | Out-Host
& (Join-Path $PSScriptRoot "build_bridge.ps1") | Out-Host
New-Item -ItemType Directory -Path $EvidenceRoot -Force | Out-Null

$Results = @()
for ($Run = 1; $Run -le $Runs; ++$Run) {
    $Started = Get-Date
    & (Join-Path $PSScriptRoot "run_probe.ps1") `
        -Profile probe-direct -EvidenceRoot $EvidenceRoot -Quiet
    if ($LASTEXITCODE -ne 0) {
        throw "direct-core cold start $Run failed with exit $LASTEXITCODE"
    }
    $WallMs = ((Get-Date) - $Started).TotalMilliseconds
    $Log = Get-ChildItem -LiteralPath $EvidenceRoot `
        -Filter "*-probe-direct.log" | Sort-Object LastWriteTimeUtc `
        -Descending | Select-Object -First 1
    if ($null -eq $Log) { throw "run $Run did not produce a probe log" }
    $CompleteLine = Get-Content -LiteralPath $Log.FullName | Where-Object {
        $_ -like '*"stage":"probe_result"*' -and
        $_ -like '*"event":"complete"*'
    } | Select-Object -Last 1
    if (-not $CompleteLine) {
        throw "run $Run has no complete probe_result: $($Log.FullName)"
    }
    $Envelope = $CompleteLine | ConvertFrom-Json
    $Value = $Envelope.value
    $TowerHp = @($Value.state.episode.crown_towers | ForEach-Object {
        [int]$_.hp
    })
    $HasSurface = [bool](Select-String -LiteralPath $Log.FullName `
        -SimpleMatch '"stage":"surface_create"' -Quiet)
    $DataTables = Get-Content -LiteralPath $Log.FullName | Where-Object {
        $_ -like '*"stage":"direct_data_tables"*'
    } | Select-Object -Last 1 | ForEach-Object {
        ($_ | ConvertFrom-Json).value
    }
    $Checks = [ordered]@{
        no_surface = -not $HasSurface
        data_tables_ready = (
            [bool]$DataTables.completed -and
            [int]$DataTables.ready_latch -eq 1 -and
            [string]$DataTables.tables -ne "0x0" -and
            [string]$DataTables.first_table -ne "0x0"
        )
        battle_ready_at_tick_zero = ([int]$Value.ready.tick -eq 0)
        advanced_exactly_100_ticks = (
            [int]$Value.step.tick_before -eq 0 -and
            [int]$Value.step.tick_after -eq 100 -and
            [int]$Value.step.stepped -eq 100
        )
        six_native_towers = (
            [int]$Value.state.entity_count -eq 6 -and
            @($Value.state.episode.crown_towers).Count -eq 6
        )
        canonical_tower_hp = (
            (Compare-Object $ExpectedTowerHp $TowerHp).Count -eq 0
        )
        coherent = [bool]$Value.state.coherent
        canonical_hash = ($Value.state.state_hash -eq $ExpectedHash)
        canonical_rng = ([uint64]$Value.state.rng_state -eq 3502570521)
        native_battle_phase = (
            [int]$Value.state.episode.native_phase.battle -eq 4
        )
    }
    $Failures = @($Checks.GetEnumerator() | Where-Object { -not $_.Value } |
        ForEach-Object { $_.Key })
    if ($Failures.Count -ne 0) {
        throw "run $Run failed checks: $($Failures -join ', ')"
    }
    $ReplayToObserveMs = [double]$Value.elapsed_ms
    $Results += [pscustomobject]@{
        run = $Run
        log = $Log.FullName
        wall_ms = [math]::Round($WallMs, 3)
        replay_to_observe_ms = $ReplayToObserveMs
        validated_path_ticks_per_second = [math]::Round(
            100000.0 / $ReplayToObserveMs, 1
        )
        state_hash = $Value.state.state_hash
        rng_state = [uint64]$Value.state.rng_state
        checks = $Checks
    }
    Write-Host ("PASS {0}/{1} hash={2} wall={3:N0}ms core-path={4:N3}ms" -f `
        $Run, $Runs, $Value.state.state_hash, $WallMs, $ReplayToObserveMs)
}

$WallValues = @($Results | ForEach-Object { [double]$_.wall_ms })
$CoreValues = @($Results | ForEach-Object {
    [double]$_.replay_to_observe_ms
})
$Summary = [ordered]@{
    schema_version = 1
    accepted = $true
    acceptance = "strict Surface-free native libg cold start and 100-tick baseline"
    runs = $Runs
    expected_hash = $ExpectedHash
    unique_hashes = @($Results.state_hash | Sort-Object -Unique)
    wall_ms = [ordered]@{
        min = ($WallValues | Measure-Object -Minimum).Minimum
        mean = ($WallValues | Measure-Object -Average).Average
        max = ($WallValues | Measure-Object -Maximum).Maximum
    }
    replay_to_observe_ms = [ordered]@{
        min = ($CoreValues | Measure-Object -Minimum).Minimum
        mean = ($CoreValues | Measure-Object -Average).Average
        max = ($CoreValues | Measure-Object -Maximum).Maximum
    }
    results = $Results
}
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$SummaryPath = Join-Path $EvidenceRoot "$Stamp-acceptance-summary.json"
$Summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $SummaryPath `
    -Encoding utf8
$Summary | ConvertTo-Json -Depth 6 | Out-Host
Write-Host "Acceptance summary: $SummaryPath"
