# Connects adb to a phone that has wireless debugging on and is already
# paired with this computer (pair once with `make pair`).
#
# adb usually connects to paired phones by itself once it sees them on the
# network. Connecting again by IP would list the same phone twice and make
# every later adb command fail with "more than one device", so this only
# connects when nothing is attached yet.
param([Parameter(Mandatory)][string]$Adb)

$attached = & $Adb devices | Select-String "`tdevice$"
if ($attached) {
    Write-Host "Already connected:"
    & $Adb devices -l
    exit 0
}

$targets = & $Adb mdns services |
    Select-String '_adb-tls-connect\._tcp\s+(\S+)' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } |
    Select-Object -Unique

if (-not $targets) {
    Write-Host "No phone is advertising wireless debugging on this network."
    Write-Host "Check that Developer options > Wireless debugging is on, and that the phone and"
    Write-Host "this computer can reach each other. Public Wi-Fi usually blocks this; use the"
    Write-Host "phone's hotspot instead."
    exit 1
}

foreach ($target in $targets) {
    & $Adb connect $target
}
& $Adb devices -l
