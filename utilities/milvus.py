from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
)


class MilvusHelper:
    def __init__(self, fields):
        try:
            self.client = MilvusClient("rag.db")
            self.collection_name = "vector_emb"

            if not self.client.has_collection(self.collection_name):
                self.create_collection(fields)
                print(f"Created collection '{self.collection_name}'.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")

            print("Connected to Milvus.")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    # Create collection
    def create_collection(self, fields):
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema._fields = fields
        index_args = {
            "field_name": "vector",
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"nlist": 2},
        }
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(**index_args)

        collection = self.client.create_collection(
            self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        return collection

    # Insert data into collection
    def upload_data(self, data):
        res = self.client.insert(collection_name=self.collection_name, data=data)

        print("Setup and data insertion into Milvus completed.")
        return res


if __name__ == "__main__":
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    ]
    milvus_client = MilvusHelper(fields)
