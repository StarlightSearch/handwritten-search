pub mod lancedb_adapter;

use crate::lancedb_adapter::{DataPoint, LanceDBAdapter, SearchResult};
use embed_anything::embed_query;
use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;
use reqwest::Client;
use serde_json::json;
use std::env;
use std::fs;
use tauri::{AppHandle, Manager};

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
async fn greet(name: &str) -> Result<String, String> {
    println!("📝 {}", name);
    Ok(name.to_string())
}

#[tauri::command]
async fn ocr_text(file_path: String) -> Result<String, String> {
    println!("🔍 Processing OCR for: {}", file_path);

    // Get API credentials
    let account_id = env::var("CLOUDFLARE_ACCOUNT_ID")
        .map_err(|_| "Missing CLOUDFLARE_ACCOUNT_ID environment variable".to_string())?;
    let api_key = env::var("API_KEY")
        .map_err(|_| "Missing API_KEY environment variable".to_string())?;

    // Read the image file - https://developers.cloudflare.com/workers-ai/models/llama-3.2-11b-vision-instruct/#Input
    let image_data = fs::read(&file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    let image_array: Vec<u8> = image_data.into_iter().collect();

    // Make API call to Cloudflare
    let client = Client::new();
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
        .map_err(|e| format!("Request failed: {}", e))?;

    // Parse response
    let result = response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let text = result["result"]["response"].as_str()
        .ok_or_else(|| format!("Failed to extract text from response"))?;

    println!("📄 Extracted text: {}", text);
    Ok(text.to_string())
}

#[tauri::command]
async fn embed_ocr_text(app: AppHandle, file_path: String) -> Result<String, String> {
    println!("🔍 Processing OCR for: {}", file_path);

    // Get API credentials
    let account_id = env::var("CLOUDFLARE_ACCOUNT_ID")
        .map_err(|_| "Missing CLOUDFLARE_ACCOUNT_ID environment variable".to_string())?;
    let api_key = env::var("API_KEY")
        .map_err(|_| "Missing API_KEY environment variable".to_string())?;

    // Read the image file
    let image_data = fs::read(&file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;
    let image_array: Vec<u8> = image_data.into_iter().collect();

    // Make API call to Cloudflare
    let client = Client::new();
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
        .map_err(|e| format!("Request failed: {}", e))?;

    // Parse response
    let result = response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let text = result["result"]["response"].as_str()
        .ok_or_else(|| format!("Failed to extract text from response"))?;

    // Generate embedding for the extracted text
    let embedder = match std::panic::catch_unwind(|| {
        Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())))
    }) {
        Ok(embedder) => embedder,
        Err(_) => return Err("Failed to create embedder".to_string()),
    };

    let embedding = embed_query(&[&text], &embedder, None).await
        .map_err(|e| format!("Failed to generate embedding: {:?}", e))?;

    if embedding.is_empty() {
        return Err("No embedding generated".to_string());
    }

    // Get the adapter from Tauri's managed state
    let adapter = app.state::<LanceDBAdapter>().inner();

    // Convert embedding to Vec<f32>
    let vector = embedding[0].embedding.to_dense().unwrap().to_vec();

    // Create data point and upsert
    let data_point = DataPoint {
        file_path: file_path.clone(),
        text: text.to_string(),
        vector,
    };

    adapter.upsert("vectors", vec![data_point]).await
        .map_err(|e| format!("Failed to upsert embedding: {:?}", e))?;

    Ok(text.to_string())
}

#[tauri::command]
async fn search_text(app: AppHandle, query: String) -> Result<Vec<SearchResult>, String> {
    // Get the adapter from Tauri's managed state
    let adapter = app.state::<LanceDBAdapter>().inner();
    let embedder = match std::panic::catch_unwind(|| {
        Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())))
    }) {
        Ok(embedder) => embedder,
        Err(e) => return Err(format!("❌ Failed to create embedder: {:?}", e)),
    };

    let embedding = match embed_query(&[&query], &embedder, None).await {
        Ok(emb) => emb,
        Err(e) => return Err(format!("❌ Failed to generate embedding: {:?}", e)),
    };

    if embedding.is_empty() {
        return Err("❌ No embedding generated".to_string());
    }

    // Convert embedding to Vec<f32>
    let vector = embedding[0].embedding.to_dense().unwrap().to_vec();

    // Perform the search
    println!("🔎 Performing vector search...");
    match adapter.search("vectors", vector, Some(10)).await {
        Ok(results) => {
            println!("✅ Search completed successfully");
            Ok(results)
        }
        Err(e) => Err(format!("❌ Search failed: {:?}", e)),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize the LanceDB adapter
    let adapter = tauri::async_runtime::block_on(async {
        // Use a relative path for the database.
        let db_uri = "./lancedb";
        println!("⏳ Initializing LanceDB Adapter at: {}", db_uri);
        let adapter_result = LanceDBAdapter::new(db_uri).await;
        match adapter_result {
            Ok(adapter) => {
                println!("✅ LanceDB Adapter initialized successfully.");

                // Ensure the default table exists
                let table_name = "vectors";
                println!("⏳ Checking for LanceDB table: '{}'", table_name);
                match adapter.db.table_names().execute().await {
                    Ok(names) => {
                        if !names.contains(&table_name.to_string()) {
                            println!("  Table '{}' not found. Creating...", table_name);
                            match adapter.create_table(table_name).await {
                                Ok(_) => {
                                    println!("  ✅ Table '{}' created successfully.", table_name)
                                }
                                Err(e) => eprintln!(
                                    "  ❌ Failed to create LanceDB table '{}': {}",
                                    table_name, e
                                ),
                            }
                        } else {
                            println!("  ✅ Table '{}' already exists.", table_name);
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to get LanceDB table names: {}", e);
                        panic!("Failed to verify table existence: {}", e);
                    }
                }
                adapter
            }
            Err(e) => {
                panic!("❌ Failed to initialize LanceDB Adapter: {}", e);
            }
        }
    });

    tauri::Builder::default()
        .manage(adapter)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, ocr_text, embed_ocr_text, search_text])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
