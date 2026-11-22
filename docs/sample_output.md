# Sample output

Captured from a run on Google Colab (T4 GPU, `Qwen2.5-1.5B-Instruct`).

## 1. Why dense retrieval?

Query: *How much vacation do I get?*

| Document | BM25 score | Dense (cosine) |
|---|---|---|
| Employees accrue 18 days of paid leave annually. | 0.00 | 0.44 |
| Unused leave up to 30 days carries over to next year. | 0.00 | 0.42 |
| The cafeteria is open from 8 am to 6 pm. | 0.00 | 0.02 |

BM25 scores every sentence 0 because none shares a word with the question. Dense embeddings still match "vacation" to "paid leave".

## 2. BM25 vs dense vs hybrid (top 3 sections)

```
error E1047 when I log in
  BM25   ['Error E1047: login t', 'Error E2001: disk qu', 'Annual leave']
  Dense  ['Error E1047: login t', 'Error E2001: disk qu', 'Password reset']
  Hybrid ['Error E1047: login t', 'Error E2001: disk qu', 'Annual leave']
When does the canteen open?
  BM25   ['Gym', 'Annual leave', 'Cafeteria']
  Dense  ['Cafeteria', 'Gym', 'Password reset']
  Hybrid ['Gym', 'Cafeteria', 'Annual leave']
```

## 3. Cross-encoder rerank

Query: *How often do I have to change my password?* → hybrid shortlist of 8, reranked:

```
['Password reset', 'Remote work', 'Remote work']
```

## 4. End-to-end answers

```
❓ How much vacation do I get? (Acme-India)
💬 Based on the information provided in the passage, you would receive 18 days of paid annual leave each year at Acme-India. This can be seen in passage [2], which states "Employees accrue 18 days of paid leave annually."
📚 ['hr_policy_india.md › Sick leave', 'hr_policy_india.md › Annual leave', 'hr_policy_india.md › Remote work']

❓ How much vacation do I get? (Acme-US)
💬 Based on the information provided in the passage, you would typically receive 15 days of paid time off (PTO) annually as part of your annual leave policy at Acme-US. This amount does not carry over from one year to the next. [1]
📚 ['hr_policy_us.md › Sick leave', 'hr_policy_us.md › Annual leave', 'hr_policy_us.md › Remote work']

❓ I'm getting error E1047, what should I do?
💬 I don't know based on the Acme documents.
📚 ['it_helpdesk.md › Error E1047: login token expired', 'it_helpdesk.md › Error E2001: disk quota exceeded', 'it_helpdesk.md › Password reset']

❓ What is the dress code on Fridays?
💬 I don't know based on the Acme documents.
📚 ['facilities.md › Cafeteria', 'hr_policy_india.md › Annual leave', 'hr_policy_india.md › Sick leave']
```

The E1047 answer is a known miss: the right chunk was ranked first, but the small model refused anyway. See the limitations section in the README.
