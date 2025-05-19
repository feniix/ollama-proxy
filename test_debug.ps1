# PowerShell script to test Ollama proxy debug endpoint
Write-Host "Testing Ollama proxy debug endpoint..." -ForegroundColor Cyan

$proxyUrl = "http://localhost:11435"
$apiKey = "llama-key"
$model = "llama3:8b"

# Test specifically with "validate llama3" payload
$payload = @{
    model = $model
    messages = @(
        @{
            role = "user"
            content = "validate llama3"
        }
    )
    tools = @(
        @{
            type = "function"
            function = @{
                name = "edit_file"
                description = "Edit a file"
                parameters = @{
                    type = "object"
                    properties = @{
                        target_file = @{
                            type = "string"
                            description = "Path to the file"
                        }
                        instructions = @{
                            type = "string"
                            description = "Instructions for the edit"
                        }
                        code_edit = @{
                            type = "string"
                            description = "The code to edit"
                        }
                    }
                    required = @("target_file", "instructions", "code_edit")
                }
            }
        }
    )
}

# Convert payload to JSON
$jsonPayload = $payload | ConvertTo-Json -Depth 10

# Send the request to debug endpoint
Write-Host "Sending request to debug endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$proxyUrl/v1/debug/chat/completions" -Method Post -Headers @{
        "Authorization" = "Bearer $apiKey"
        "Content-Type" = "application/json"
    } -Body $jsonPayload -ErrorAction Stop
    
    # Display the response
    Write-Host "Debug endpoint response:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error with debug endpoint:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody" -ForegroundColor Red
    }
}

# Now try the same request to normal endpoint
Write-Host "`nSending same request to normal endpoint..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$proxyUrl/v1/chat/completions" -Method Post -Headers @{
        "Authorization" = "Bearer $apiKey"
        "Content-Type" = "application/json"
    } -Body $jsonPayload -ErrorAction Stop
    
    # Display the response
    Write-Host "Regular endpoint response:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error with regular endpoint:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody" -ForegroundColor Red
    }
}

Write-Host "`nTests completed. Check the proxy logs for detailed information." -ForegroundColor Cyan 