from pymilvus import CollectionSchema, Collection, MilvusClient, connections
from pymilvus import DataType, FieldSchema

class MilvusHelper:
    def __init__(self, fields):
        try:
            self.client = MilvusClient("rag.db")
            self.collection_name = "Milvus_uploading"
            self.collection = self.create_collection(fields)

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
        index_args = {"field_name": "embeddings",
                      "index_type": "HNSW",
                      "metric_type": "L2",
                      "params": {"nlist": 2}
                    }
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(**index_args)

        collection = self.client.create_collection(self.collection_name, schema=schema, index_params=index_params, consistency_level="Strong")
        return collection

    # Insert data into collection
    def upload_data(self, data):
        res = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )
        # res = collection.insert(data)
        # collection.flush()
        print("Setup and data insertion into Milvus completed.")
        return res

if __name__ == '__main__':
    urls = ["https://1.com", "https://2.com", "https://3.com", "https://4.com", "https://5.com", "https://6.com", "https://7.com", "https://8.com", "https://9.com", "https://10.com"]
    import torch
    embeddings = torch.randn((len(urls), 384))

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="urls", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    milvus_client = MilvusHelper(fields)
    data = [{"id": i, "urls": urls[i], "embeddings": embeddings[i]} for i in range(len(urls))]

    uploaded = milvus_client.upload_data(data)
    breakpoint()