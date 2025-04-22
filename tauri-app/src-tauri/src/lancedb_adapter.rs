use arrow_array::types::Float32Type;
use arrow_array::{
    FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connection::Connection;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, DistanceType, Result, Table as LanceDbTable};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

const VECTOR_DIMENSION: i32 = 512;

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub file_path: String,
    pub text: String,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub file_path: String,
    pub text: String,
    pub score: f32, // LanceDB returns _distance
}

pub struct LanceDBAdapter {
    pub db: Connection,
}

impl LanceDBAdapter {
    pub async fn new(uri: &str) -> Result<Self> {
        let db = connect(uri).execute().await?;
        Ok(Self { db })
    }

    pub async fn create_table(&self, table_name: &str) -> Result<LanceDbTable> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    VECTOR_DIMENSION,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
            Field::new("file_path", DataType::Utf8, true),
        ]));

        let table = self
            .db
            .create_empty_table(table_name, schema)
            .execute()
            .await?;
        Ok(table)
    }

    async fn convert_data_to_arrow(
        data_points: Vec<DataPoint>,
    ) -> Result<(RecordBatch, Arc<Schema>)> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    VECTOR_DIMENSION,
                ),
                true,
            ),
            Field::new("text", DataType::Utf8, true),
            Field::new("file_path", DataType::Utf8, true),
        ]));

        let total_rows = data_points.len();

        let ids: Vec<String> = (0..total_rows)
            .map(|_| Uuid::new_v4().to_string())
            .collect();
        let texts: Vec<String> = data_points.iter().map(|dp| dp.text.clone()).collect();
        let file_paths: Vec<String> = data_points.iter().map(|dp| dp.file_path.clone()).collect();
        let vectors = data_points.iter().map(|dp| dp.vector.clone());

        let id_array = Arc::new(StringArray::from(ids));
        let text_array = Arc::new(StringArray::from(texts));
        let file_path_array = Arc::new(StringArray::from(file_paths));
        let vector_array = Arc::new(
            FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vectors.map(|v| Some(v.into_iter().map(Some))),
                VECTOR_DIMENSION,
            ),
        );

        let record_batch = RecordBatch::try_new(
            schema.clone(),
            vec![id_array, vector_array, text_array, file_path_array],
        )?;

        Ok((record_batch, schema))
    }

    pub async fn upsert(&self, table_name: &str, data_points: Vec<DataPoint>) -> Result<()> {
        println!("📝 Number of data points to upsert: {}", data_points.len());
        let table = self.db.open_table(table_name).execute().await?;

        println!("🔄 Converting data to Arrow format...");
        let (data, schema) = Self::convert_data_to_arrow(data_points).await?;
        println!("📊 Record batch schema: {:?}", schema);

        println!("📦 Creating record batch iterator...");
        let reader = Box::new(RecordBatchIterator::new(
            vec![data].into_iter().map(Ok),
            schema,
        ));

        println!("💾 Adding data to table...");
        table.add(reader).execute().await?;
        println!("✅ Data successfully added to table");

        Ok(())
    }

    pub async fn search(
        &self,
        table_name: &str,
        query_vector: Vec<f32>,
        limit: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        let table = self.db.open_table(table_name).execute().await?;
        let limit = limit.unwrap_or(20);

        let results = table
            .vector_search(query_vector)?
            .distance_type(DistanceType::Cosine)
            .limit(limit)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        let mut search_results = Vec::new();
        for batch in results {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let texts = batch
                .column_by_name("text")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let file_paths = batch
                .column_by_name("file_path")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let distances = batch
                .column_by_name("_distance")
                .unwrap()
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();

            for i in 0..batch.num_rows() {
                search_results.push(SearchResult {
                    id: ids.value(i).to_string(),
                    text: texts.value(i).to_string(),
                    file_path: file_paths.value(i).to_string(),
                    score: distances.value(i),
                });
            }
        }

        Ok(search_results)
    }
}
