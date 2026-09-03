$inputJson = [Console]::In.ReadToEnd()
$event = $inputJson | ConvertFrom-Json

$project = if ($event.cwd) { $event.cwd } else { "Unknown project" }
$message = "VS Code agent run finished.`nProject: $project"

Invoke-RestMethod `
  -Uri "https://api.telegram.org/bot$env:TELEGRAM_BOT_TOKEN/sendMessage" `
  -Method Post `
  -Body @{
    chat_id = $env:TELEGRAM_CHAT_ID
    text    = $message
  } | Out-Null

Write-Output '{"continue":true}'
