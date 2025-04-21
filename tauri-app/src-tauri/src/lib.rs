// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

use std::fs;
use std::env;
use reqwest::Client;
use serde_json::json;

#[tauri::command]
async fn embed_ocr_text(file_path: String) -> Result<String, String> {
    println!("🔍 Starting OCR process for file: {}", file_path);

    // Get API credentials from environment variables
    println!("🔑 Fetching API credentials...");
    let account_id = env::var("CLOUDFLARE_ACCOUNT_ID")
        .expect("❌ CLOUDFLARE_ACCOUNT_ID not found in environment variables");
    let api_key = env::var("API_KEY")
        .expect("❌ API_KEY not found in environment variables");
    println!("✅ API credentials retrieved successfully");

    // Read the image file
    println!("📂 Reading image file...");
    let image_data = fs::read(&file_path)
        .map_err(|e| format!("❌ Failed to read file: {}", e))?;
    println!("✅ Image file read successfully ({} bytes)", image_data.len());

    // Convert to array of numbers (like in Svelte)
    println!("🔄 Converting image to array of numbers...");
    let image_array: Vec<u8> = image_data.into_iter().collect();
    println!("✅ Image converted to array ({} elements)", image_array.len());

    // Create HTTP client
    println!("🌐 Preparing HTTP client...");
    let client = Client::new();

    // Make API call to Cloudflare
    println!("🚀 Sending request to Cloudflare API...");
    let response = client
        .post(format!("https://api.cloudflare.com/client/v4/accounts/{}/ai/run/@cf/meta/llama-3.2-11b-vision-instruct", account_id))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&json!({
            "prompt": "Please extract the text from the provided image. Do not include any formatting, explanations, or descriptions. Only provide the plain text contained in the image.",
            "image": image_array
        }))
        .send()
        .await
        .map_err(|e| format!("❌ Failed to send request: {}", e))?;
    println!("✅ Request sent successfully (Status: {})", response.status());

    // Parse response
    println!("📥 Parsing API response...");
    let result = response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| format!("❌ Failed to parse response: {}", e))?;
    
    // Print the full JSON response for debugging
    println!("📄 Full JSON response:");
    println!("{}", serde_json::to_string_pretty(&result).unwrap_or_else(|_| "Failed to pretty print JSON".to_string()));
    
    println!("✅ Response parsed successfully");

    // Extract the text from the response
    println!("🔍 Extracting text from response...");
    println!("🔍 Checking result structure...");
    println!("- Has 'result' field: {}", result.get("result").is_some());
    if let Some(result_field) = result.get("result") {
        println!("- Has 'response' field: {}", result_field.get("response").is_some());
        if let Some(response_field) = result_field.get("response") {
            println!("- Response type: {}", response_field);
        }
    }

    let text = result["result"]["response"]
        .as_str()
        .ok_or_else(|| {
            let error_msg = format!(
                "❌ Failed to extract text from response. Response structure: {}",
                serde_json::to_string(&result).unwrap_or_else(|_| "Failed to serialize response".to_string())
            );
            println!("{}", error_msg);
            error_msg
        })?;
    
    println!("✅ Text extracted successfully");
    println!("📝 Extracted text length: {} characters", text.len());
    println!("📝 Extracted text: {}", text);

    Ok(text.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, embed_ocr_text])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
