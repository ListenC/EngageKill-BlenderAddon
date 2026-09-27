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

# Create ZIP.
# Compress-Archive must not be used here: on Windows PowerShell 5.1 it writes no
# folder entries at all, so a package that contains subdirectories (cache/,
# presets/) makes the add-on updater's extraction fail with a missing-directory
# error. Folders are written explicitly instead, parents before their contents,
# so even updater builds that only mkdir on trailing-slash entries can install it.
Write-Host "Creating archive: $zipName ..."

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Add-TreeToZip {
    param(
        [System.IO.Compression.ZipArchive]$Zip,
        [string]$Dir,
        [string]$Prefix
    )
    foreach ($sub in Get-ChildItem -LiteralPath $Dir -Directory | Sort-Object Name) {
        if ($sub.Name -eq "__pycache__") { continue }
        $Zip.CreateEntry("$Prefix/$($sub.Name)/") | Out-Null
        Add-TreeToZip -Zip $Zip -Dir $sub.FullName -Prefix "$Prefix/$($sub.Name)"
    }
    foreach ($file in Get-ChildItem -LiteralPath $Dir -File | Sort-Object Name) {
        $entry = $Zip.CreateEntry("$Prefix/$($file.Name)")
        $target = $entry.Open()
        $source = [System.IO.File]::OpenRead($file.FullName)
        try { $source.CopyTo($target) } finally {
            $source.Dispose()
            $target.Dispose()
        }
    }
}

$zipPath = Join-Path (Get-Location).Path $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$zipStream = [System.IO.File]::Open($zipPath, [System.IO.FileMode]::CreateNew)
$zip = New-Object System.IO.Compression.ZipArchive(
    $zipStream, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $zip.CreateEntry("$folder/") | Out-Null
    Add-TreeToZip -Zip $zip -Dir (Resolve-Path $folder).Path -Prefix $folder
} finally {
    $zip.Dispose()
    $zipStream.Dispose()
}

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
