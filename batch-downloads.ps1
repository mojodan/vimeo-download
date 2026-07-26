# Activate the venv at C:\Users\dhaye\repos\vimeo if it isn't already active
$venvRoot = Split-Path -Parent $PSScriptRoot
if ($env:VIRTUAL_ENV -ne $venvRoot) {
    & (Join-Path $venvRoot "Scripts\Activate.ps1")
}

python vimeo_download.py --keep-audio -m medium https://vimeo.com/1212143123/d7599fc5fc --output-dir "G:\OpenTrader\coaching-webinar\20260722" -desc
