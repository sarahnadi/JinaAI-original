from docarray import DocumentArray, Document
import numpy as np
import json
from jina import Flow

#  Build a DocumentArray
# We loop over json data to concatenate the meaningful fields (title, textual attributes, short description, name, supplier)
# that we later want to use for our embeddings
da = DocumentArray()

with open('icecat-products-w_price-19k-20201127.json') as f:
    jdocs = json.load(f)
for doc in jdocs:
    d = Document(text=" ".join([doc[item] for item in doc if item.startswith("title") 
    or item.startswith("attr_t")
    or item.startswith("short_description") 
    or item.startswith("name")
    or item.startswith("supplier")]), tags=doc)
    da.append(d)

 
print(da.summary())
print(10*"*********")
print(da[0])

### Build a Flow
#### Our Flow consists of an encoder and an indexer. 
# The encoder uses the TransformerTorchEncoder to generate embeddings, 
# the indexer indexes the embeddings together with some metadata (supplier, price, attr_t_product_type, attr_product_colour) 
# that can be used to filter the data later on.

# f = Flow()

f = Flow().add(name='encoder', uses='jinahub://TransformerTorchEncoder/latest', install_requirements=True).add(uses='jinahub://AnnLiteIndexer/latest', install_requirements=True, uses_with={'columns': [('supplier', 'str'), ('price', 'float'), ('attr_t_product_type', 'str'), ('attr_t_product_colour', 'str')], 'n_dim': 768})

with f:
  pass
##here we only choose the first 1000 products from the original JSON dataset, to be indexed (only for demo to make indexing faster)
# we can comment this line if we want to index the whole dataset
print("*"*100)
print("This is the total length of the document array before choosing the first 1000 elements to be indexed",len(da))
da = da[1:1000]

with f:
  da = f.index(da)
  print(da[0])
  print(type(da))
## pickle f into a file 
  
## another script to load the pickle and run the query script
  
print(100*"===")
# store the above into a pickle - Python object serialization

query = Document(text='laptop case')


####
with f:
  results = f.search(query)
  
print(results[0].matches[0].tags)
with f:
  results = f.search(Document(text='candle'))
print("="*40)
print(results[0].matches[0].tags)

# query = Document(text='laptop case')
# query = Document(text='laptop')
# with f:
#   results = f.search(query)
# print(results[0].matches[0].tags)
