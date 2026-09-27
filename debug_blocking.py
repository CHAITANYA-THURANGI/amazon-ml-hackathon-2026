import pandas as pd

queries = [
    ("S1-106407869", "Vision Partners", "US"),
    ("S1-156285671", "Team Ecole", "France"),
    ("S1-909865979", "Cure Seafood", "US"),
    ("S1-280204013", "Om Constructions", "India"),
    ("S1-680637447", "Siliguri Media", "India"),
]

for s1_id, q, country in queries:
    print("=" * 60)
    print(f"SEARCHING FOR {s1_id}: '{q}' in {country}")
    
    # Search S2
    s2_hits = []
    for chunk in pd.read_csv("dataset/test/test_source2.tsv", sep="\t", chunksize=500_000):
        h = chunk[chunk["business_name"].fillna("").str.contains(q, case=False, regex=False)]
        if len(h):
            s2_hits.append(h)
        if sum(len(x) for x in s2_hits) >= 2:
            break
    if s2_hits:
        for _, r in pd.concat(s2_hits).head(2).iterrows():
            print("  S2 HIT:", r["entity_id"], "|", r["country"], "|", r["business_name"], "|", r["business_address"])
    else:
        print("  S2: NO SUBSTRING HIT")

    # Search S3
    s3_hits = []
    for chunk in pd.read_csv("dataset/test/test_source3.tsv", sep="\t", chunksize=500_000):
        h = chunk[chunk["business_name"].fillna("").str.contains(q, case=False, regex=False)]
        if len(h):
            s3_hits.append(h)
        if sum(len(x) for x in s3_hits) >= 2:
            break
    if s3_hits:
        for _, r in pd.concat(s3_hits).head(2).iterrows():
            print("  S3 HIT:", r["entity_id"], "|", r["country"], "|", r["business_name"], "|", r["business_address"])
    else:
        print("  S3: NO SUBSTRING HIT")
