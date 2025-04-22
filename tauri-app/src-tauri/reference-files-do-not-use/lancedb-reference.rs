use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use embed_anything::embeddings::embed::EmbedData;

use crate::SearchResponse;
use arrow_array::{RecordBatch, RecordBatchIterator};
use arrow_schema::DataType;
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::vector::IvfPqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, DistanceType, Result, Table as LanceDbTable};
use std::path::Path;
use uuid::Uuid;

const DIM: usize = 512;
pub async fn create_lance_db(uri: &str) -> Result<Connection> {
    let db = connect(uri).execute().await?;
    Ok(db)
}

pub async fn create_table(db: &mut Connection, table_name: &str) -> Result<LanceDbTable> {
    let schema = Arc::new(lancedb::arrow::arrow_schema::Schema::new(vec![
        lancedb::arrow::arrow_schema::Field::new(
            "id",
            lancedb::arrow::arrow_schema::DataType::Utf8,
            true,
        ),
        lancedb::arrow::arrow_schema::Field::new(
            "file_name",
            lancedb::arrow::arrow_schema::DataType::Utf8,
            false,
        ),
        lancedb::arrow::arrow_schema::Field::new(
            "page_number",
            lancedb::arrow::arrow_schema::DataType::Utf8,
            true,
        ),
        lancedb::arrow::arrow_schema::Field::new(
            "workspace_path",
            lancedb::arrow::arrow_schema::DataType::Utf8,
            false,
        ),
        lancedb::arrow::arrow_schema::Field::new(
            "workspace_name",
            lancedb::arrow::arrow_schema::DataType::Utf8,
            false,
        ),
        lancedb::arrow::arrow_schema::Field::new(
            "text",
            lancedb::arrow::arrow_schema::DataType::Utf8,
            false,
        ),
        lancedb::arrow::arrow_schema::Field::new(
            "vector",
            lancedb::arrow::arrow_schema::DataType::FixedSizeList(
                Arc::new(lancedb::arrow::arrow_schema::Field::new(
                    "item",
                    lancedb::arrow::arrow_schema::DataType::Float32,
                    true,
                )),
                DIM as i32,
            ),
            true,
        ),
    ]));

    let mut table = db.create_empty_table(table_name, schema).execute().await?;
    create_index(&mut table).await?;
    Ok(table)
}

pub async fn _create_vector_index(table: &mut LanceDbTable) -> Result<()> {
    // Get the number of rows in the table
    let count = table.count_rows(None).await?;

    // Only create indices if we have enough data
    if count >= 256 {
        // Minimum threshold for creating IVF-PQ index
        table
            .create_index(
                &["vector"],
                Index::IvfPq(
                    IvfPqIndexBuilder::default()
                        .distance_type(DistanceType::Cosine)
                        .num_partitions(50)
                        .num_sub_vectors(16),
                ),
            )
            .execute()
            .await?;

        println!("Indices created");
    } else {
        println!(
            "Skipping index creation - not enough rows in table (need at least 256, got {count})"
        );
    }
    Ok(())
}

pub async fn convert_data_to_arrow(
    workspace_name: &str,
    workspace_path: &str,
    vectors: Vec<EmbedData>,
) -> Result<(arrow_array::RecordBatch, Arc<arrow_schema::Schema>)> {
    // Define the schema
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", DataType::Utf8, true),
        arrow_schema::Field::new("file_name", DataType::Utf8, false),
        arrow_schema::Field::new("page_number", DataType::Utf8, true),
        arrow_schema::Field::new("workspace_path", DataType::Utf8, false),
        arrow_schema::Field::new("workspace_name", DataType::Utf8, false),
        arrow_schema::Field::new("text", DataType::Utf8, false),
        arrow_schema::Field::new(
            "vector",
            arrow_schema::DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::Float32,
                    true,
                )),
                DIM as i32,
            ),
            true,
        ),
    ]));

    // Prepare arrays for each field
    let ids: Vec<String> = vectors.iter().map(|_| Uuid::new_v4().to_string()).collect();
    let file_names: Vec<String> = vectors
        .iter()
        .map(|e| {
            e.metadata.as_ref().unwrap()["file_name"]
                .clone()
                .replace("\\\\?\\", "")
        })
        .collect();
    let texts: Vec<String> = vectors
        .iter()
        .map(|e| e.text.as_ref().unwrap().clone())
        .collect();
    let page_numbers: Vec<Option<String>> = vectors
        .iter()
        .map(|e| {
            e.metadata
                .as_ref()
                .unwrap()
                .get("page_number")
                .map(|s| s.to_string())
        })
        .collect();
    let embeddings = vectors.iter().map(|e| e.embedding.to_dense().unwrap());

    let id_array = Arc::new(arrow_array::StringArray::from(ids));
    let file_name_array = Arc::new(arrow_array::StringArray::from(file_names));
    let page_number_array = Arc::new(arrow_array::StringArray::from(page_numbers));
    let workspace_path_array = Arc::new(arrow_array::StringArray::from(vec![
        workspace_path;
        vectors.len()
    ]));
    let workspace_name_array = Arc::new(arrow_array::StringArray::from(vec![
        workspace_name;
        vectors.len()
    ]));
    let text_array = Arc::new(arrow_array::StringArray::from(texts));
    let vector_array = Arc::new(arrow_array::FixedSizeListArray::from_iter_primitive::<
        arrow_array::types::Float32Type,
        _,
        _,
    >(
        embeddings.map(|v| Some(v.into_iter().map(Some))),
        DIM as i32,
    ));

    // Return the RecordBatch directly instead of wrapping in an iterator
    let record_batch = arrow_array::RecordBatch::try_new(
        schema.clone(),
        vec![
            id_array,
            file_name_array,
            page_number_array,
            workspace_path_array,
            workspace_name_array,
            text_array,
            vector_array,
        ],
    )
    .unwrap();

    Ok((record_batch, schema))
}

pub async fn create_index(table: &mut LanceDbTable) -> Result<()> {
    table
        .create_index(&["text"], Index::FTS(FtsIndexBuilder::default()))
        .execute()
        .await?;

    Ok(())
}
pub async fn upsert(
    table: &mut LanceDbTable,
    workspace_name: &str,
    workspace_path: &str,
    vectors: Vec<EmbedData>,
) -> Result<()> {
    let (data, schema) = convert_data_to_arrow(workspace_name, workspace_path, vectors).await?;
    let reader = Box::new(RecordBatchIterator::new(
        vec![data].into_iter().map(Ok),
        schema,
    ));
    table.add(reader).execute().await?;
    Ok(())
}

async fn _get_all_workspaces(table: &mut LanceDbTable) -> Result<Vec<String>> {
    let result = table
        .query()
        .select(lancedb::query::Select::Columns(vec![
            "workspace_name".to_string()
        ]))
        .execute()
        .await?;

    let batch = result.try_collect::<Vec<_>>().await?;
    let workspaces: Vec<String> = batch
        .iter()
        .flat_map(|rb| {
            rb.column(0)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    Ok(workspaces)
}

async fn _get_files_in_vector_db(
    table: &mut LanceDbTable,
    workspace_name: &str,
) -> Result<Vec<String>> {
    let result = table
        .query()
        .select(lancedb::query::Select::Columns(vec!["text".to_string()]))
        .only_if(format!("workspace_name == '{}'", workspace_name).as_str())
        .limit(1000)
        .execute()
        .await?;
    let batch = result.try_collect::<Vec<_>>().await?;
    println!("batch: {:?}", batch);
    let file_names: Vec<String> = batch
        .iter()
        .flat_map(|rb| {
            rb.column(0)
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .iter()
                .flatten()
                .map(|s| s.to_string())
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut file = File::create("file_names.md").unwrap();
    for file_name in &file_names {
        file.write_all(file_name.as_bytes()).unwrap();
        file.write_all("\n-----------------------------------\n".as_bytes())
            .unwrap();
    }
    Ok(file_names)
}

pub async fn delete_workspace(table: &LanceDbTable, workspace_name: &str) -> Result<()> {
    table
        .delete(format!("workspace_name == '{}'", workspace_name).as_str())
        .await?;
    Ok(())
}

pub async fn search(
    table: &LanceDbTable,
    workspace_name: &str,
    query_point: Vec<f32>,
    search_query: &str,
    limit: Option<usize>,
    context_files: Vec<String>,
    rerank: bool,
) -> Result<Vec<SearchResponse>> {
    let limit = limit.unwrap_or(30);

    // Create the base workspace filter
    let mut filter = format!("workspace_name == '{}'", workspace_name);

    // Add context files filter if provided
    if !context_files.is_empty() {
        let file_paths = context_files
            .iter()
            .map(|s| format!("'{}'", s.replace("'", "''"))) // Escape single quotes
            .collect::<Vec<_>>()
            .join(",");
        filter = format!("{} AND file_name IN ({})", filter, file_paths);
    }

    // Get vector search results with filter
    let vector_search_result: Vec<RecordBatch> = table
        .vector_search(query_point)?
        .only_if(&filter)
        .distance_type(DistanceType::Cosine)
        .limit(30)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    // Get text search results with the same filter
    let text_search_result = match table
        .query()
        .full_text_search(FullTextSearchQuery::new(search_query.to_string()))
        .limit(limit)
        .only_if(&filter)
        .execute()
        .await
    {
        Ok(result) => match result.try_collect::<Vec<_>>().await {
            Ok(batches) => batches,
            Err(e) => {
                println!("Error collecting text search results: {:?}", e);
                Vec::new()
            }
        },
        Err(e) => {
            println!("Error executing text search: {:?}", e);
            Vec::new()
        }
    };
    let mut deduplicated_vector_results = Vec::new();
    let mut deduplicated_text_results = Vec::new();
    let mut seen_vector_results: HashSet<String> = HashSet::new();
    let mut seen_text_results: HashSet<String> = HashSet::new();

    // Process vector search results
    for batch in vector_search_result {
        let texts = batch
            .column_by_name("text")
            .unwrap_or_else(|| panic!("text column not found"));

        for i in 0..batch.num_rows() {
            let text = texts
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .value(i)
                .to_string();

            if seen_vector_results.insert(text) {
                deduplicated_vector_results.push(batch.slice(i, 1));
            }
        }
    }

    // Process text search results
    for batch in text_search_result {
        let texts = batch
            .column_by_name("text")
            .unwrap_or_else(|| panic!("text column not found"));

        for i in 0..batch.num_rows() {
            let text = texts
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .value(i)
                .to_string();

            if seen_text_results.insert(text) {
                deduplicated_text_results.push(batch.slice(i, 1));
            }
        }
    }

    let mut combined_results: Vec<SearchResponse> = Vec::new();
    let mut rrf_scores: HashMap<String, f32> = HashMap::new();
    const K: f32 = 60.0;

    // Calculate RRF scores for vector results
    for (batch_index, record_batch) in deduplicated_vector_results.iter().enumerate() {
        process_batch_for_rrf_scores(batch_index, record_batch, &mut rrf_scores, K);
    }

    // Calculate RRF scores for text results
    for (batch_index, record_batch) in deduplicated_text_results.iter().enumerate() {
        process_batch_for_rrf_scores(batch_index, record_batch, &mut rrf_scores, K);
    }

    // Process results
    for record_batch in deduplicated_vector_results
        .iter()
        .chain(deduplicated_text_results.iter())
    {
        process_batch_for_results(record_batch, &rrf_scores, &mut combined_results);
    }

    // Sort by RRF score descending
    combined_results.sort_by(|a, b| {
        b.score
            .parse::<f32>()
            .unwrap_or(0.0)
            .partial_cmp(&a.score.parse::<f32>().unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if rerank {
        let texts: Vec<&str> = combined_results.iter().map(|r| r.text.as_str()).collect();
        let reranker_model = embed_anything::reranker::model::Reranker::new(
            "jinaai/jina-reranker-v1-turbo-en",
            None,
            embed_anything::Dtype::F16,
        )
        .unwrap();
        let reranker_scores = reranker_model
            .compute_scores(vec![search_query], texts, 1)
            .unwrap();
        let scores = reranker_scores.first().unwrap();

        // Update scores in combined_results with reranker scores
        for (i, result) in combined_results.iter_mut().enumerate() {
            result.score = scores[i].to_string();
        }

        // Sort by reranker scores descending
        combined_results.sort_by(|a, b| {
            b.score
                .parse::<f32>()
                .unwrap_or(0.0)
                .partial_cmp(&a.score.parse::<f32>().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let combined_results = combined_results
        .into_iter()
        .filter(|r| r.text.chars().count() > 300)
        .collect::<Vec<_>>();
    Ok(combined_results)
}

fn process_batch_for_rrf_scores(
    batch_index: usize,
    record_batch: &RecordBatch,
    rrf_scores: &mut HashMap<String, f32>,
    k: f32,
) {
    for row_index in 0..record_batch.num_rows() {
        let id = record_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(row_index)
            .to_string();

        let rank = (batch_index * record_batch.num_rows() + row_index + 1) as f32;
        *rrf_scores.entry(id).or_insert(0.0) += 1.0 / (rank + k);
    }
}
fn process_batch_for_results(
    record_batch: &RecordBatch,
    rrf_scores: &HashMap<String, f32>,
    combined_results: &mut Vec<SearchResponse>,
) {
    for row_index in 0..record_batch.num_rows() {
        let text = record_batch
            .column_by_name("text")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(row_index)
            .to_string();

        // Skip if we've already seen this text content
        if combined_results.iter().any(|r| r.text == text) {
            continue;
        }

        let id = record_batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap()
            .value(row_index)
            .to_string();

        let page_number = record_batch.column_by_name("page_number").and_then(|col| {
            col.as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .map(|arr| arr.value(row_index).to_string())
        });

        let response = SearchResponse {
            file_name: Path::new(
                &record_batch
                    .column_by_name("file_name")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .unwrap()
                    .value(row_index)
                    .to_string(),
            )
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string(),
            file_path: record_batch
                .column_by_name("file_name")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .unwrap()
                .value(row_index)
                .to_string(),
            text,
            page_number,
            score: rrf_scores.get(&id).unwrap_or(&0.0).to_string(),
            id,
        };
        combined_results.push(response);
    }
}
