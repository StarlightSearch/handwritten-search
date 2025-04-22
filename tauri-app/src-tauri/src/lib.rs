pub mod lancedb_adapter;

use std::fs;
use std::env;
use reqwest::Client;
use serde_json::json;
use embed_anything::embed_query;
use embed_anything::embeddings::embed::{Embedder, TextEmbedder};
use embed_anything::embeddings::local::jina::JinaEmbedder;
use crate::lancedb_adapter::{LanceDBAdapter, DataPoint, SearchResult};
use tauri::{Manager, AppHandle};

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
async fn search_text(app: AppHandle, query: String) -> Result<Vec<SearchResult>, String> {
    println!("🔍 Starting search for query: {}", query);
    
    // Get the adapter from Tauri's managed state
    let adapter = app.state::<LanceDBAdapter>().inner();

    // Generate embedding for the query
    println!("⏳ Generating embedding for query...");
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
        },
        Err(e) => Err(format!("❌ Search failed: {:?}", e)),
    }
}

#[tauri::command]
async fn embed_ocr_text(app: AppHandle, file_path: String) -> Result<String, String> {
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
    
    println!("📝 Extracted text length: {} characters", text.len());

    // Generate embedding for the extracted text
    println!("⏳ Generating embedding for extracted text...");
    let embedder = match std::panic::catch_unwind(|| {
        Embedder::Text(TextEmbedder::Jina(Box::new(JinaEmbedder::default())))
    }) {
        Ok(embedder) => embedder,
        Err(e) => return Err(format!("❌ Failed to create embedder: {:?}", e)),
    };

    let embedding = match embed_query(&[&text], &embedder, None).await {
        Ok(emb) => emb,
        Err(e) => return Err(format!("❌ Failed to generate embedding: {:?}", e)),
    };

    if embedding.is_empty() {
        return Err("❌ No embedding generated".to_string());
    }

    println!("📊 Embedding dimension: {}", embedding[0].embedding.to_dense().unwrap().len());

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

    println!("💾 Upserting embedding to LanceDB...");
    match adapter.upsert("vectors", vec![data_point]).await {
        Ok(_) => {
            println!("✅ Embedding upserted successfully");
            Ok(text.to_string())
        },
        Err(e) => {
            println!("❌ Failed to upsert embedding: {:?}", e);
            Err(format!("❌ Failed to upsert embedding: {:?}", e))
        },
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
                                Ok(_) => println!("  ✅ Table '{}' created successfully.", table_name),
                                Err(e) => eprintln!("  ❌ Failed to create LanceDB table '{}': {}", table_name, e),
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
        .invoke_handler(tauri::generate_handler![greet, embed_ocr_text, search_text])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
