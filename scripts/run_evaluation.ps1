param(
    [Parameter(Mandatory = $true)]
    [string]$BaseUrl,

    [Parameter(Mandatory = $true)]
    [string]$Jwt,

    [Parameter(Mandatory = $true)]
    [string]$ChatId,

    [Parameter(Mandatory = $true)]
    [string]$PdfPath,

    [string]$QuestionsFile = "testdata/control_questions_seed_40.json",
    [int]$TopK = 5,
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"

$authHeaders = @{
    Authorization = "Bearer $Jwt"
}

$jsonHeaders = @{
    Authorization = "Bearer $Jwt"
    "Content-Type" = "application/json"
}

Write-Host "1) Upload + ingest document..."
$uploadForm = @{
    chat_id = $ChatId
    file    = Get-Item $PdfPath
}
$uploadResponse = Invoke-RestMethod -Method Post -Uri "$BaseUrl/documents/upload" -Headers $authHeaders -Form $uploadForm
$uploadResponse | ConvertTo-Json -Depth 6 | Write-Host

Write-Host "2) Batch import control questions..."
$questionPayload = Get-Content -Path $QuestionsFile -Raw
$batchResponse = Invoke-RestMethod -Method Post -Uri "$BaseUrl/chats/$ChatId/test-questions/batch" -Headers $jsonHeaders -Body $questionPayload
$batchResponse | ConvertTo-Json -Depth 6 | Write-Host

Write-Host "3) Start evaluation run..."
$runBody = @{
    chat_id = $ChatId
    top_k   = $TopK
    model   = $Model
} | ConvertTo-Json
$runResponse = Invoke-RestMethod -Method Post -Uri "$BaseUrl/evaluations/runs" -Headers $jsonHeaders -Body $runBody
$runResponse | ConvertTo-Json -Depth 10 | Write-Host

$runId = $runResponse.id
if (-not $runId) {
    throw "run id is missing in response"
}

Write-Host "4) Fetch run metrics..."
$metricsResponse = Invoke-RestMethod -Method Get -Uri "$BaseUrl/evaluations/runs/$runId/metrics" -Headers $authHeaders
$metricsResponse | ConvertTo-Json -Depth 6 | Write-Host

Write-Host "5) Compare with baseline keyword search..."
$baselineResponse = Invoke-RestMethod -Method Get -Uri "$BaseUrl/evaluations/runs/$runId/baseline?limit=1" -Headers $authHeaders
$baselineResponse | ConvertTo-Json -Depth 6 | Write-Host

Write-Host "Done."
