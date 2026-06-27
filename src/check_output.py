import csv

rows = list(csv.DictReader(open("data/output/submission.csv", encoding="utf-8")))
print(f"Total rows: {len(rows)}")
print(f"Score range: {rows[0]['score']} -> {rows[-1]['score']}")
print()
print("Top 10:")
for r in rows[:10]:
    cid = r["candidate_id"]
    score = r["score"]
    reason = r["reasoning"][:120]
    print(f"  #{r['rank']:>3}  {cid}  score={score}")
    print(f"        {reason}")
    print()
