# Removes repeated <meta-data> lines from the generated AndroidManifest.xml.
# The react-native-google-mobile-ads config plugin adds its meta-data entries
# again on each `expo prebuild`, and duplicate entries can break manifest
# merging.
param([string]$Manifest = "$PSScriptRoot/../android/app/src/main/AndroidManifest.xml")

$path = (Resolve-Path $Manifest).Path
$text = [IO.File]::ReadAllText($path)
$newline = if ($text.Contains("`r`n")) { "`r`n" } else { "`n" }

$seen = [Collections.Generic.HashSet[string]]::new()
$removed = 0
$kept = foreach ($line in $text -split "\r?\n") {
    if ($line.Trim().StartsWith('<meta-data') -and -not $seen.Add($line.Trim())) {
        $removed++
        continue
    }
    $line
}

if ($removed -gt 0) {
    [IO.File]::WriteAllText($path, ($kept -join $newline), [Text.UTF8Encoding]::new($false))
}
Write-Host "Removed $removed duplicate meta-data line(s) from AndroidManifest.xml"
