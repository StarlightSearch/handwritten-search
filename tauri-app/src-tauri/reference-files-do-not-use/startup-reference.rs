use chrono::TimeDelta;
use diesel::RunQueryDsl;
use diesel::SqliteConnection;
use hf_hub::api::Progress;
use lancedb::Table as LanceDbTable;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tauri::async_runtime::Mutex;
use tauri::Emitter;
use tauri::Manager;

use crate::config::AppConfig;
use crate::database;
use crate::database::model::Workspace;
use crate::database::operations::delete_workspace;
use crate::indexing;
use crate::indexing::operations::embed_one_file;
use crate::schema::workspaces;
use crate::vectordb;
use crate::EmbeddingModel;
use embed_anything::config::SplittingStrategy;
use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::EmbedderBuilder;

#[derive(Clone)]
struct MyProgress {
    current: usize,
    total: usize,
    app_handle: tauri::AppHandle,
    message: String,
}

impl Progress for MyProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        self.total = size;
        self.current = 0;
    }

    fn update(&mut self, size: usize) {
        self.current += size;
        self.app_handle
            .emit(
                "download_progress",
                (self.message.clone(), self.current, self.total),
            )
            .unwrap();
    }

    fn finish(&mut self) {
        self.app_handle
            .emit(
                "download_progress",
                (self.message.clone(), self.current, self.total),
            )
            .unwrap();
    }
}

pub async fn setup_database(app_dir: &str) -> Arc<Mutex<SqliteConnection>> {
    let db_start = Instant::now();
    let connection = Arc::new(Mutex::new(database::operations::establish_connection(
        app_dir,
    )));
    println!(
        "📦 Database connection established: {:?}",
        db_start.elapsed()
    );
    connection
}

pub async fn setup_vector_db(app_dir: &str) -> Arc<Mutex<LanceDbTable>> {
    let vectordb_start = Instant::now();
    let table = Arc::new(Mutex::new({
        let mut db = vectordb::lance::create_lance_db(app_dir).await.unwrap();
        let table_names: Vec<String> = db.table_names().execute().await.unwrap();
        if !table_names.contains(&"test".to_string()) {
            let table = vectordb::lance::create_table(&mut db, "test")
                .await
                .unwrap();
            println!(
                "📊 Vector DB tables created: {:?}",
                db.table_names().execute().await.unwrap()
            );
            table
                .optimize(lancedb::table::OptimizeAction::All)
                .await
                .unwrap();
            table
        } else {
            db.open_table("test").execute().await.unwrap()
        }
    }));
    println!(
        "🔍 Vector DB setup complete: {:?}",
        vectordb_start.elapsed()
    );
    table
}

pub async fn setup_embedding_model(
    app_config: &AppConfig,
    app_handle: &tauri::AppHandle,
) -> EmbeddingModel {
    let model_start = Instant::now();

    std::env::set_var(
        "HF_HOME",
        app_handle
            .app_handle()
            .path()
            .app_data_dir()
            .unwrap()
            .to_str()
            .unwrap(),
    );

    let dense_model = Arc::new(
        EmbedderBuilder::new()
            .model_architecture("jina")
            .model_id(Some(&app_config.embedding_model.model_name))
            .path_in_repo(Some(&app_config.embedding_model.path_in_repo))
            .from_pretrained_onnx()
            .unwrap(),
    );

    println!("🧠 Embedding model loaded: {:?}", model_start.elapsed());

    let config = TextEmbedConfig::default()
        .with_chunk_size(app_config.chunk_size, Some(app_config.chunk_overlap as f32))
        .with_batch_size(1)
        .with_splitting_strategy(SplittingStrategy::Sentence);

    EmbeddingModel {
        dense_model,
        config,
    }
}

/// Check if the model appears to be completely downloaded
fn is_model_valid(model_path: &Path) -> bool {
    log::debug!("Validating model at: {}", model_path.display());

    // Basic existence check
    if !model_path.exists() {
        log::debug!("Model path does not exist");
        return false;
    }

    // Check for essential directories
    let required_dirs = ["blobs", "refs", "snapshots"];
    for dir in required_dirs {
        if !model_path.join(dir).exists() {
            log::debug!("Missing required directory: {}", dir);
            return false;
        }
    }

    // Check for refs/main pointer file
    let pointer_path = model_path.join("refs").join("main");
    if !pointer_path.exists() {
        log::debug!("Missing refs/main pointer file");
        return false;
    }

    true
}

pub async fn download_models(
    app_config: &AppConfig,
    app_handle: &tauri::AppHandle,
) -> Result<(), anyhow::Error> {
    log::info!("🚀 Starting model downloads check...");
    let start = Instant::now();

    let cache_dir = app_handle.app_handle().path().app_data_dir()?.join("hub");
    log::debug!("Using cache directory: {}", cache_dir.display());

    let api_repo = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.clone().into())
        .build()?;

    // Models to download
    let models = [
        (&app_config.reranker_model, "Reranker Model"),
        (&app_config.embedding_model, "Embedding Model"),
    ];

    for (model_config, model_name) in models {
        let model_path = cache_dir.join(format!(
            "models--{}",
            model_config.model_name.replace("/", "--")
        ));
        log::info!("📦 Checking {}: {}", model_name, model_path.display());

        // Clean up potentially corrupted downloads
        if model_path.exists() && !is_model_valid(&model_path) {
            log::warn!(
                "🧹 Found incomplete model, cleaning up: {}",
                model_path.display()
            );
            if let Err(e) = std::fs::remove_dir_all(&model_path) {
                log::error!("Failed to clean up corrupted model: {}", e);
            }
        }

        if !is_model_valid(&model_path) {
            log::info!(
                "⏳ {} not found or incomplete, starting download...",
                model_name
            );
            let download_start = Instant::now();

            let progress = MyProgress {
                current: 0,
                total: 0,
                app_handle: app_handle.clone(),
                message: format!("Downloading {}", model_name),
            };

            let model = api_repo.model(model_config.model_name.clone());
            match model.download_with_progress(model_config.path_in_repo.as_str(), progress) {
                Ok(_) => {
                    // Verify download was successful
                    if !is_model_valid(&model_path) {
                        log::error!("❌ {} download appears incomplete", model_name);
                        return Err(anyhow::anyhow!("Model download validation failed"));
                    }
                    log::info!(
                        "✅ {} downloaded successfully in {:?}",
                        model_name,
                        download_start.elapsed()
                    );
                }
                Err(e) => {
                    log::error!("❌ Failed to download {}: {}", model_name, e);
                    // Clean up failed download
                    if model_path.exists() {
                        let _ = std::fs::remove_dir_all(&model_path);
                    }
                    return Err(e.into());
                }
            }
        } else {
            log::info!("✅ {} already present and valid", model_name);
        }
    }

    log::info!("🎉 Model downloads completed in {:?}", start.elapsed());
    Ok(())
}

pub async fn clean_invalid_workspaces(
    connection: &mut SqliteConnection,
    table: &LanceDbTable,
) -> Result<(), anyhow::Error> {
    let all_workspaces = workspaces::table.load::<Workspace>(connection)?;

    for workspace in all_workspaces {
        if !std::path::Path::new(&workspace.workspace_path).exists() {
            delete_workspace(connection, &workspace.workspace_name)?;
            vectordb::lance::delete_workspace(table, &workspace.workspace_name).await?;
        }
    }

    Ok(())
}

pub async fn index_all_workspaces(
    connection: Arc<Mutex<SqliteConnection>>,
    table: Arc<Mutex<LanceDbTable>>,
    app_handle: &tauri::AppHandle,
    app_config: &AppConfig,
) -> Result<(), anyhow::Error> {
    let connection = Arc::clone(&connection);
    let table = Arc::clone(&table);

    let embedding_model = setup_embedding_model(app_config, app_handle).await;

    let all_workspaces = workspaces::table.load::<Workspace>(&mut *connection.lock().await)?;

    for workspace in all_workspaces {
        let extension_regex = indexing::operations::get_extension_regex();

        let all_files_in_workspace_directory =
            indexing::operations::get_all_file_paths(&workspace.workspace_path)
                .await?
                .into_iter()
                .filter(|file_path| extension_regex.is_match(file_path))
                .collect::<Vec<_>>();

        database::operations::update_all_file_paths(
            &mut *connection.lock().await,
            &workspace.workspace_name,
            &all_files_in_workspace_directory,
        )?;

        let file_paths = database::operations::get_embedded_file_paths(
            &mut *connection.lock().await,
            &workspace.workspace_name,
        )?;
        let files_to_delete = file_paths
            .iter()
            .filter(|file_path| !all_files_in_workspace_directory.contains(file_path))
            .collect::<Vec<_>>();

        for file_path in files_to_delete {
            indexing::operations::remove_embedded_file(
                file_path,
                Arc::clone(&connection),
                Arc::clone(&table),
                app_handle,
            )
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        }

        let files_to_embed = all_files_in_workspace_directory
            .iter()
            .filter(|file_path| !file_paths.contains(file_path))
            .filter(|file_path| extension_regex.is_match(file_path))
            .collect::<Vec<_>>();
        let _all_file_paths = workspace.get_all_file_paths();
        println!("🔍 Files to embed: {:?}", files_to_embed);

        for file_path in files_to_embed {
            embed_one_file(
                file_path,
                Some(workspace.workspace_name.clone()),
                &embedding_model,
                Arc::clone(&connection),
                Arc::clone(&table),
                app_handle,
                app_config,
            )
            .await
            .map_err(|e| anyhow::anyhow!(e))?;
        }
    }
    let table_guard = table.lock().await;
    table_guard
        .optimize(lancedb::table::OptimizeAction::Prune {
            older_than: Some(TimeDelta::from_std(Duration::from_secs(30))?),
            delete_unverified: Some(true),
            error_if_tagged_old_versions: Some(true),
        })
        .await?;
    // table_guard
    //     .optimize(lancedb::table::OptimizeAction::All)
    //     .await?;

    println!(
        "🔢 Number of rows in table: {}",
        table_guard.count_rows(None).await?
    );

    Ok(())
}
