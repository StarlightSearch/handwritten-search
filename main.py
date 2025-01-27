from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.lancedb_adapter import LanceDBAdapter
import uvicorn

app = FastAPI()
adapter = LanceDBAdapter()

class CollectionDelete(BaseModel):
    collection_name: str

@app.delete("/collections/delete")
async def delete_collection(request: CollectionDelete):
    try:
        adapter.delete_index("")  # Empty string as we don't use the index_name parameter
        return {"message": "All data cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
