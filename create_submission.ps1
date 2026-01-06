# Script to create submission-ready zip file
# Excludes development files but keeps all necessary code and data

$projectRoot = Get-Location
$submissionDir = "Data_Analysis_AI_App_Submission"
$zipFile = "Data_Analysis_AI_App_Submission.zip"

# Remove old submission files if they exist
if (Test-Path $submissionDir) {
    Remove-Item -Recurse -Force $submissionDir
}
if (Test-Path $zipFile) {
    Remove-Item -Force $zipFile
}

# Create submission directory
New-Item -ItemType Directory -Path $submissionDir | Out-Null

# Files and directories to include
$includeItems = @(
    "app",
    "src",
    "data/sample_statements",
    "tests",
    "docs",
    "README.md",
    "requirements.txt",
    ".env.example",
    "pytest.ini",
    "generate_graph_diagram.py"
)

# Copy each item
foreach ($item in $includeItems) {
    $sourcePath = Join-Path $projectRoot $item
    $destPath = Join-Path $submissionDir $item
    
    if (Test-Path $sourcePath) {
        # Create parent directory if needed
        $parentDir = Split-Path $destPath -Parent
        if (-not (Test-Path $parentDir)) {
            New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
        }
        
        # Copy the item
        if (Test-Path $sourcePath -PathType Container) {
            Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
        } else {
            Copy-Item -Path $sourcePath -Destination $destPath -Force
        }
        Write-Host "Copied: $item" -ForegroundColor Green
    } else {
        Write-Host "Skipped (not found): $item" -ForegroundColor Yellow
    }
}

# Clean up __pycache__ and other dev files from copied directory
Get-ChildItem -Path $submissionDir -Recurse -Include "__pycache__","*.pyc","*.pyo",".pytest_cache" | Remove-Item -Recurse -Force
Write-Host "`nCleaned up Python cache files" -ForegroundColor Green

# Create zip file
Write-Host "`nCreating zip file..." -ForegroundColor Cyan
Compress-Archive -Path $submissionDir -DestinationPath $zipFile -Force

# Show results
$zipSize = (Get-Item $zipFile).Length / 1MB
Write-Host "`n=== Submission Package Created ===" -ForegroundColor Green
Write-Host "Location: $projectRoot\$zipFile" -ForegroundColor White
Write-Host "Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor White

# List contents
Write-Host "`nContents of submission package:" -ForegroundColor Cyan
Get-ChildItem -Path $submissionDir -Recurse -File | Select-Object -ExpandProperty FullName | ForEach-Object {
    $relativePath = $_.Replace("$projectRoot\$submissionDir\", "")
    Write-Host "  - $relativePath" -ForegroundColor Gray
}

Write-Host "`nSubmission ready! Upload: $zipFile" -ForegroundColor Green
