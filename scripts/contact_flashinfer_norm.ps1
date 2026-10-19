param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [string]$BodyPath,
    [switch]$Post
)
$ErrorActionPreference = 'Stop'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'Never'
if (Test-Path -LiteralPath $OutDir) { throw 'Use a fresh output directory.' }
New-Item -ItemType Directory -Path $OutDir | Out-Null
# Keep credentials in process memory only; never emit or save helper output.
$credentialLines = "protocol=https`nhost=github.com`n`n" | git -c credential.interactive=never credential fill 2>$null
if ($LASTEXITCODE -ne 0) { throw 'No usable noninteractive GitHub credential; no message sent.' }
$passwordLine = $credentialLines | Where-Object { $_.StartsWith('password=') } | Select-Object -First 1
if (-not $passwordLine) { throw 'No usable GitHub credential; no message sent.' }
$headers = @{
    Authorization = 'Bearer ' + $passwordLine.Substring(9)
    Accept = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
    'User-Agent' = 'cloud-inference-project'
}
$credentialLines = $null
$passwordLine = $null
$utf8 = New-Object System.Text.UTF8Encoding($false)
try {
    $account = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers
    if ($account.login -ne 'kerry0520') { throw 'Unexpected authenticated account; no message sent.' }
    $issueUrl = 'https://api.github.com/repos/flashinfer-ai/flashinfer/issues/5355'
    $issue = Invoke-RestMethod -Uri $issueUrl -Headers $headers
    $comments = @(Invoke-RestMethod -Uri ($issueUrl + '/comments?per_page=100') -Headers $headers)
    [IO.File]::WriteAllText((Join-Path $OutDir 'issue.json'), ($issue | ConvertTo-Json -Depth 30), $utf8)
    [IO.File]::WriteAllText((Join-Path $OutDir 'comments.json'), (ConvertTo-Json -InputObject $comments -Depth 30), $utf8)
    Write-Output ('Account=' + $account.login + ' State=' + $issue.state + ' Comments=' + $issue.comments)
    if (-not $Post) {
        $query = [Uri]::EscapeDataString('repo:flashinfer-ai/flashinfer is:pr "5355"')
        $related = Invoke-RestMethod -Uri ('https://api.github.com/search/issues?q=' + $query + '&per_page=30') -Headers $headers
        [IO.File]::WriteAllText((Join-Path $OutDir 'related-prs.json'), ($related | ConvertTo-Json -Depth 30), $utf8)
        $related.items | ForEach-Object { Write-Output ($_.state + ' ' + $_.html_url + ' ' + $_.title) }
    }
    if ($Post) {
        if ($issue.state -ne 'open') { throw 'Issue is closed; no message sent.' }
        if ($issue.comments -gt 100) { throw 'Comment pagination needs review before posting.' }
        $body = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $BodyPath), [Text.Encoding]::UTF8)
        $existing = $comments | Where-Object { $_.user.login -eq 'kerry0520' -and $_.body -eq $body } | Select-Object -First 1
        if ($existing) {
            Write-Output ('Already posted: ' + $existing.html_url)
        } else {
            $payload = ConvertTo-Json -InputObject @{body=$body} -Compress
            $result = Invoke-RestMethod -Method Post -Uri ($issueUrl + '/comments') -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
            [IO.File]::WriteAllText((Join-Path $OutDir 'posted.json'), ($result | ConvertTo-Json -Depth 30), $utf8)
            Write-Output ('Posted: ' + $result.html_url)
        }
    }
} finally {
    $headers.Authorization = $null
}
