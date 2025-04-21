use std::fs;
use std::env;
use reqwest::Client;
use serde_json::json;
use embed_anything::embed_query;
use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
async fn greet(name: &str) -> Result<String, String> {
    println!("🔍 Starting greet function with name: {}", name);
    
    println!("⏳ Attempting to create Jina embedder...");
    let embedder = match std::panic::catch_unwind(|| {
        println!("  Creating Jina embedder object...");
        let result = Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())));
        println!("  Jina embedder created successfully");
        result
    }) {
        Ok(embedder) => embedder,
        Err(e) => {
            let error_msg = format!("❌ Failed to create embedder: {:?}", e);
            println!("{}", error_msg);
            return Err(error_msg);
        }
    };
    println!("✅ Embedder created successfully");
    
    println!("⏳ Attempting to generate embedding...");
    let embedding_result = match std::panic::catch_unwind(|| {
        println!("  Will call embed_query next...");
        true // Just to indicate we reached this point without panicking
    }) {
        Ok(_) => {
            println!("  Calling embed_query...");
            match embed_query(&[name], &embedder, None).await {
                Ok(embedding) => {
                    println!("  Embedding generated successfully");
                    Ok(embedding)
                },
                Err(e) => {
                    let error_msg = format!("❌ embed_query returned an error: {:?}", e);
                    println!("{}", error_msg);
                    Err(error_msg)
                }
            }
        },
        Err(e) => {
            let error_msg = format!("❌ Panic before embed_query call: {:?}", e);
            println!("{}", error_msg);
            Err(error_msg)
        }
    };
    
    // Process the result
    let embedding = match embedding_result {
        Ok(emb) => emb,
        Err(e) => return Err(e),
    };
    
    println!("✅ Embedding generated successfully");
    println!("📊 Embedding data count: {}", embedding.len());
    
    if !embedding.is_empty() {
        println!("💡 First embedding info: {:?}", embedding[0]);
    }
    
    let result = format!("Hello, {}! Your embedding has been generated with {} vectors", name, embedding.len());
    println!("📝 Final result: {}", result);
    
    Ok(result)
}

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
