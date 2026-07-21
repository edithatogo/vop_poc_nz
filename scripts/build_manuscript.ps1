param(
  [string]$Root = (Get-Location).Path,
  [string]$Main = "manuscript/jss_submission.tex"
)
$ErrorActionPreference = "Stop"
$compiler = "C:\Users\60217257\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py"
$output = Join-Path $Root "manuscript/build"
uv run python $compiler (Join-Path $Root $Main) --compiler tectonic --output-directory $output --json
if (-not (Test-Path (Join-Path $output "jss_submission.pdf"))) {
  throw "LaTeX build did not produce manuscript/build/jss_submission.pdf; install TeX Live with biber and rerun."
}
