param([Parameter(Mandatory=$true)][string]$OutDir,
      [string[]]$Repos = @('vllm-project/vllm','sgl-project/sglang','flashinfer-ai/flashinfer','InternLM/lmdeploy','ModelTC/lightllm','ggml-org/llama.cpp'))
$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'Never'
if (Test-Path -LiteralPath $OutDir) { throw 'Use a new evidence directory.' }
New-Item -ItemType Directory -Path $OutDir | Out-Null
$lines = "protocol=https`nhost=github.com`n`n" | git -c credential.interactive=never credential fill 2>$null
if ($LASTEXITCODE -ne 0) { throw 'No noninteractive GitHub credential.' }
$secret = $lines | Where-Object { $_.StartsWith('password=') } | Select-Object -First 1
if (-not $secret) { throw 'No usable GitHub credential.' }
$headers = @{Authorization='Bearer ' + $secret.Substring(9);Accept='application/vnd.github+json';'User-Agent'='cloud-inference-readonly-research'}
$lines = $null
$secret = $null
$utf8 = New-Object System.Text.UTF8Encoding($false)
$manifest = @()
try {
    foreach ($repo in $repos) {
        $query = 'repo:' + $repo + ' is:issue is:open no:assignee comments:0'
        $url = 'https://api.github.com/search/issues?q=' + [Uri]::EscapeDataString($query) + '&sort=created&order=desc&per_page=20'
        $name = $repo.Replace('/','-') + '.json'
        try {
            $payload = Invoke-RestMethod -Uri $url -Headers $headers
        } catch {
            # Stop rather than retry a rate-limited request. Never serialize headers.
            $failure = @{repo=$repo;url=$url;failed_utc=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message;response_body=$_.ErrorDetails.Message}
            [IO.File]::WriteAllText((Join-Path $OutDir 'failure.json'),($failure | ConvertTo-Json -Depth 5),$utf8)
            throw
        }
        [IO.File]::WriteAllText((Join-Path $OutDir $name),($payload | ConvertTo-Json -Depth 50),$utf8)
        $manifest += @{repo=$repo;query=$query;url=$url;file=$name;fetched_utc=[DateTime]::UtcNow.ToString('o');total=$payload.total_count}
        Write-Output ('REPO ' + $repo + ' total=' + $payload.total_count)
        foreach ($item in $payload.items) { Write-Output ($item.number.ToString() + ' | ' + $item.title) }
        Start-Sleep -Seconds 10
    }
} finally {
    $headers.Authorization = $null
    [IO.File]::WriteAllText((Join-Path $OutDir 'manifest.json'),(ConvertTo-Json -InputObject $manifest -Depth 8),$utf8)
}
