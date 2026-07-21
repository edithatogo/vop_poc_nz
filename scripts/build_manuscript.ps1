param(
  [string]$Root = (Get-Location).Path,
  [string]$Main = "manuscript/jss_submission.tex"
)
$ErrorActionPreference = "Stop"
$manuscript = Join-Path $Root "manuscript"
$latexmkCommand = Get-Command latexmk -ErrorAction SilentlyContinue
$latexmkPath = if ($latexmkCommand) { $latexmkCommand.Source } else { $null }
if (-not $latexmkPath) {
  $miktex = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\latexmk.exe"
  if (Test-Path $miktex) { $latexmkPath = $miktex }
}
if (-not $latexmkPath) { throw "latexmk was not found; install TeX Live or MiKTeX." }
$texBin = Split-Path -Parent $latexmkPath
$originalPath = $env:Path
$env:Path = "$texBin;$originalPath"

Push-Location $manuscript
try {
  foreach ($build in @("build", "build-jss")) {
    New-Item -ItemType Directory -Force $build | Out-Null
    Copy-Item -LiteralPath "ref.bib" -Destination (Join-Path $build "ref.bib") -Force
    Copy-Item -LiteralPath "jss.bst" -Destination (Join-Path $build "jss.bst") -Force
  }
  & $latexmkPath -g -pdf -interaction=nonstopmode -file-line-error -outdir=build "jss_submission.tex"
  if ($LASTEXITCODE -ne 0) { throw "arXiv/preprint LaTeX build failed" }
  & $latexmkPath -g -pdf -interaction=nonstopmode -file-line-error -outdir=build-jss "jss_journal.tex"
  if ($LASTEXITCODE -ne 0) { throw "JSS LaTeX build failed" }
} finally {
  Pop-Location
  $env:Path = $originalPath
}
foreach ($pdf in @("manuscript/build/jss_submission.pdf", "manuscript/build-jss/jss_journal.pdf")) {
  if (-not (Test-Path (Join-Path $Root $pdf))) { throw "LaTeX build did not produce $pdf" }
}
