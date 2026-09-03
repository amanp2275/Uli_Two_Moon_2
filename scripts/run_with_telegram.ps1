param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("real_nvp", "transformer")]
  [string]$Model
)

$ErrorActionPreference = "Continue"
$started = Get-Date
$exitCode = 0

try {
  & python "experiments/run_individual.py" --model $Model
  $exitCode = $LASTEXITCODE
}
catch {
  $exitCode = 1
}

$status = if ($exitCode -eq 0) { "completed successfully" } else { "failed (exit code $exitCode)" }
$duration = ((Get-Date) - $started).ToString("hh\:mm\:ss")
$token = [Environment]::GetEnvironmentVariable("TELEGRAM_BOT_TOKEN", "User")
$chatId = [Environment]::GetEnvironmentVariable("TELEGRAM_CHAT_ID", "User")

if (-not [string]::IsNullOrWhiteSpace($token) -and -not [string]::IsNullOrWhiteSpace($chatId)) {
  Invoke-RestMethod `
    -Uri "https://api.telegram.org/bot$token/sendMessage" `
    -Method Post `
    -Body @{
      chat_id = $chatId
      text = "RealNVP/Transformer run ($Model) $status.`nDuration: $duration"
    } | Out-Null
}
else {
  Write-Warning "Telegram environment variables are missing; no notification was sent."
}

exit $exitCode
