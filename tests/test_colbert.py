from colbert import Searcher
searcher = Searcher(
    index="/home/peter/.cache/colbert/indexes/default",
    collection="/home/peter/.cache/colbert/collections/default.tsv",
    checkpoint="colbert-ir/colbertv2.0"
)
results = searcher.search("test query", k=3)
print(results)