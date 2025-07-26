# Ensure UTF8 output
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$folder = "apply_blendshapes"
$license = "LICENSE"
$initPath = Join-Path $folder "__init__.py"

if (-not (Test-Path $initPath)) {
    Write-Host "Error: __init__.py not found in $folder"
    exit 1
}

# Extract version from bl_info block
$inside = $false
$version = $null
Get-Content $initPath | ForEach-Object {
    if ($_ -match '^\s*bl_info\s*=\s*{') { $inside = $true; return }
    if ($inside -and $_ -match '^\s*}') { $inside = $false; return }
    if ($inside -and $_ -match '"version"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)') {
        $m = [regex]::Match($_, '"version"\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)')
        $version = "$($m.Groups[1].Value).$($m.Groups[2].Value).$($m.Groups[3].Value)"
    }
}

if (-not $version) {
    Write-Host "Error: Failed to parse version from bl_info"
    exit 1
}
Write-Host "Version detected: $version"

$zipName = "${folder}_$version.zip"

# Copy LICENSE into plugin folder
if (-not (Test-Path $license)) {
    Write-Host "Error: LICENSE file not found."
    exit 1
}
Write-Host "Copying LICENSE into $folder..."
Copy-Item $license -Destination $folder -Force

# Create ZIP
Write-Host "Creating archive: $zipName ..."
Compress-Archive -Path $folder -DestinationPath $zipName -Force

if (Test-Path $zipName) {
    Write-Host "Archive created successfully."
} else {
    Write-Host "Error: ZIP creation failed."
    Remove-Item -Path (Join-Path $folder (Split-Path $license)) -ErrorAction SilentlyContinue
    exit 1
}

# Remove LICENSE from folder
Write-Host "Removing LICENSE from within $folder..."
Remove-Item (Join-Path $folder $license) -Force

Write-Host "Done."
exit 0
