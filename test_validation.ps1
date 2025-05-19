# PowerShell script to test Ollama proxy tool validation
Write-Host "Testing Ollama proxy tool validation..." -ForegroundColor Cyan

$proxyUrl = "http://localhost:11435"
$apiKey = "llama-key"
$model = "llama3:8b"

# Validation request payload
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
                        content = @{
                            type = "string"
                            description = "Content to write"
                        }
                    }
                    required = @("target_file", "content")
                }
            }
        }
    )
}

# Convert payload to JSON
$jsonPayload = $payload | ConvertTo-Json -Depth 10

# Send the request
Write-Host "Sending validation request..." -ForegroundColor Cyan
try {
    $response = Invoke-RestMethod -Uri "$proxyUrl/v1/chat/completions" -Method Post -Headers @{
        "Authorization" = "Bearer $apiKey"
        "Content-Type" = "application/json"
    } -Body $jsonPayload -ErrorAction Stop
    
    # Display the response
    Write-Host "Validation successful!" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Cyan
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "Error during validation:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody" -ForegroundColor Red
    }
}

# Now test a regular completion with tools
Write-Host "`nTesting regular completion with tools..." -ForegroundColor Cyan

$payload = @{
    model = $model
    messages = @(
        @{
            role = "user"
            content = "Create a simple Python function to print hello world"
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
                        content = @{
                            type = "string"
                            description = "Content to write"
                        }
                    }
                    required = @("target_file", "content")
                }
            }
        }
    )
}

# Convert payload to JSON
$jsonPayload = $payload | ConvertTo-Json -Depth 10

# Send the request
try {
    $response = Invoke-RestMethod -Uri "$proxyUrl/v1/chat/completions" -Method Post -Headers @{
        "Authorization" = "Bearer $apiKey"
        "Content-Type" = "application/json"
    } -Body $jsonPayload -ErrorAction Stop
    
    # Display the response
    Write-Host "Regular completion successful!" -ForegroundColor Green
    Write-Host "Response contains tool calls: $($response.choices[0].message.tool_calls -ne $null)" -ForegroundColor Cyan
    
    # Show the first part of the tool call if it exists
    if ($response.choices[0].message.tool_calls) {
        $toolCall = $response.choices[0].message.tool_calls[0]
        Write-Host "Tool call name: $($toolCall.function.name)" -ForegroundColor Cyan
        $argumentsJson = $toolCall.function.arguments | ConvertFrom-Json
        Write-Host "Function arguments preview: $($argumentsJson | ConvertTo-Json -Depth 1)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "Error during regular completion:" -ForegroundColor Red
    Write-Host $_.Exception.Message
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody" -ForegroundColor Red
    }
}

Write-Host "`nTests completed. You can now try using this model in Cursor." -ForegroundColor Cyan 