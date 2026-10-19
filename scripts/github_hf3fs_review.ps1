param([Parameter(Mandatory=$true)][string]$OutDir, [switch]$Fork,
      [switch]$Publish, [switch]$HumanReviewed, [string]$BodyPath, [switch]$Status)
$ErrorActionPreference='Stop'
$env:GIT_TERMINAL_PROMPT='0'
$env:GCM_INTERACTIVE='Never'
if(Test-Path -LiteralPath $OutDir){throw 'Use a new result directory'}
New-Item -ItemType Directory -Path $OutDir | Out-Null
$credentialLines="protocol=https`nhost=github.com`n`n" | git -c credential.interactive=never credential fill 2>$null
if($LASTEXITCODE -ne 0){throw 'No noninteractive credential'}
$passwordLine=$credentialLines | Where-Object {$_.StartsWith('password=')} | Select-Object -First 1
if(-not $passwordLine){throw 'No GitHub credential'}
$headers=@{Authorization='Bearer '+$passwordLine.Substring(9);Accept='application/vnd.github+json';'User-Agent'='hf3fs-review'}
$credentialLines=$null
$passwordLine=$null
$utf8=New-Object System.Text.UTF8Encoding($false)
function Save-Response($name,$value){[IO.File]::WriteAllText((Join-Path $OutDir $name),(ConvertTo-Json -InputObject $value -Depth 50),$utf8)}
try{
    $account=Invoke-RestMethod https://api.github.com/user -Headers $headers
    if($account.login -ne 'kerry0520'){throw 'Unexpected GitHub account'}
    if(([int]$Fork.IsPresent+[int]$Publish.IsPresent+[int]$Status.IsPresent) -gt 1){throw 'Select one mode'}
    if($Publish){
        if(-not $HumanReviewed){throw 'Submitting human review and own test run required'}
        $body=[IO.File]::ReadAllText((Resolve-Path -LiteralPath $BodyPath),[Text.Encoding]::UTF8)
        $head='kerry0520:fix/hf3fs-miss-page-budget'
        $existing=Invoke-RestMethod ('https://api.github.com/repos/vllm-project/vllm/pulls?state=open&head='+[Uri]::EscapeDataString($head)) -Headers $headers
        if(@($existing).Count -gt 0){Save-Response 'existing-pr.json' $existing; Write-Output $existing[0].html_url}
        else{
            $payload=@{title='[Bugfix][KV Connector] Allocate HF3FS pages only for distinct misses';head=$head;base='main';body=$body;draft=$false} | ConvertTo-Json -Depth 6
            $pr=Invoke-RestMethod -Method Post https://api.github.com/repos/vllm-project/vllm/pulls -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($payload))
            Save-Response 'published-pr.json' $pr
            Write-Output $pr.html_url
        }
    }elseif($Fork){
        try{$repo=Invoke-RestMethod https://api.github.com/repos/kerry0520/vllm -Headers $headers}
        catch{if([int]$_.Exception.Response.StatusCode -ne 404){throw}}
        if(-not $repo){$repo=Invoke-RestMethod -Method Post https://api.github.com/repos/vllm-project/vllm/forks -Headers $headers -ContentType 'application/json' -Body '{"default_branch_only":true}'}
        if(-not $repo.fork -or $repo.parent.full_name -ne 'vllm-project/vllm'){throw 'Unexpected fork parent'}
        Save-Response 'fork.json' $repo
        Write-Output $repo.html_url
    }elseif($Status){
        $pr=Invoke-RestMethod https://api.github.com/repos/vllm-project/vllm/pulls/62393 -Headers $headers
        Save-Response 'pr.json' $pr
        $checks=Invoke-RestMethod ('https://api.github.com/repos/vllm-project/vllm/commits/'+$pr.head.sha+'/check-runs?per_page=100') -Headers $headers
        Save-Response 'checks.json' $checks
        Write-Output ('PR='+$pr.number+' state='+$pr.state+' head='+$pr.head.sha+' labels='+($pr.labels.name -join ','))
        foreach($check in $checks.check_runs){
            Write-Output ($check.name+' '+$check.status+' '+$check.conclusion)
            if($check.name -eq 'pre-run-check' -and $check.conclusion -eq 'failure'){
                $annotations=Invoke-RestMethod $check.output.annotations_url -Headers $headers
                Save-Response ('annotations-'+$check.id+'.json') @($annotations)
                foreach($annotation in $annotations){Write-Output $annotation.message}
            }
        }
    }else{
        $issue=Invoke-RestMethod https://api.github.com/repos/vllm-project/vllm/issues/62389 -Headers $headers
        Save-Response 'issue.json' $issue
        $comments=Invoke-RestMethod 'https://api.github.com/repos/vllm-project/vllm/issues/62389/comments?per_page=100' -Headers $headers
        Save-Response 'comments.json' @($comments)
        Write-Output ('Issue state='+$issue.state+' comments='+$issue.comments)
        foreach($query in @('repo:vllm-project/vllm is:pr is:open "62389" in:body','repo:vllm-project/vllm is:pr is:open HF3FS')){
            $name=if($query.Contains('in:body')){'related-number.json'}else{'related-area.json'}
            $result=Invoke-RestMethod ('https://api.github.com/search/issues?q='+[Uri]::EscapeDataString($query)+'&per_page=100') -Headers $headers
            Save-Response $name $result
            Write-Output ($name+' total='+$result.total_count)
            foreach($item in $result.items){Write-Output ($item.html_url+' '+$item.title)}
            Start-Sleep -Seconds 10
        }
    }
}catch{
    Save-Response 'failure.json' @{error=$_.Exception.Message;response_body=$_.ErrorDetails.Message}
    throw
}finally{$headers.Authorization=$null}
