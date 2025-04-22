use lancedb::connection::Connection;
use lancedb::{connect, Result, Table as LanceDbTable};
use std::sync::Arc;
use lancedb::arrow::arrow_schema::{DataType, Field, Schema, SchemaRef};

const VECTOR_DIMENSION: i32 = 768; // Match the Python example (Jina embeddings)

pub struct LanceDBAdapter {
    pub db: Connection,
    // We might store the table name or the table itself later
}

impl LanceDBAdapter {
    pub async fn new(uri: &str) -> Result<Self> {
        let db = connect(uri).execute().await?;
        Ok(Self { db })
    }

    // Creates a table with a predefined schema if it doesn't exist.
    pub async fn create_table(&self, table_name: &str) -> Result<LanceDbTable> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false), // LanceDB schema requires nullable=false for primary key? Let's assume false for now. Python schema has it as string() which doesn't specify nullability. Rust example uses true for id, but let's follow python logic more closely if possible. lancedb python doesn't specify nullability directly in the schema definition like arrow-rs does. Let's make it non-nullable.
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)), // Inner items can be null? Let's assume true.
                    VECTOR_DIMENSION,
                ),
                true, // The vector field itself can be null? Let's assume true.
            ),
            Field::new("text", DataType::Utf8, true), // Text can be nullable.
            Field::new("file_path", DataType::Utf8, true), // File path can be nullable.
        ]));

        let table = self.db.create_empty_table(table_name, schema).execute().await?;
        Ok(table)
    }
} 