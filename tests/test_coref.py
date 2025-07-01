from fastcoref import FCoref

model = FCoref(device="cpu", model_name_or_path="biu-nlp/f-coref")
text = "Alice went to her room where she opened the window and found that the sun was shining brightly. Bob was outside and saw her. He waved at her."
clusters = model.predict(texts=[text])
print(clusters[0].get_clusters(as_strings=True))