# pageindex-open

Truly open pageindex RAG package


This package was inspired by [PageIndex](https://github.com/VectifyAI/PageIndex). I took inspiration from the concepts outlined and came up with my own implementation. I was not satisfied with the package as examples focus on the SaaS part of things.

This package works by simply converting your PDFs into a tree then the most relevent section is decided and used. This contrasts with chuncking where similarity is compared using embeddings.

## Why?

- 🧠 **Reasoning-backed:** AI  routes and answers using structured context, not just similarity.  
- ⚡ **Contrast to RAG:** Traditional RAG retrieves random chunks by embedding similarity: here, relevance is hierarchical and precise.  
- 🌳 **Tree-structured:** Sections, subsections, and headings preserved: your document is understood, not just searched.  
- 🔢 **Top-K retrieval:** Combine multiple relevant sections for richer answers, avoiding “partial context” problems.  
- ✂️ **Text-on-demand:** Only the node text is used, no bloated storage or duplication.  
- 💾 **Persistent cache:** Markdown + tree saved separately: queries can be re-run without touching the PDF.  
- 📄 **Markdown source:** Human-readable, diffable, and editable: not a black-box blob of vectors.  
- 🔄 **Reusable & update-friendly:** Swap LLMs, add PDFs, or refresh sections without breaking the index.  
- 📦 **Clean Python API:** `build_index()`, `query()`, `load_index()`: intuitive for devs.  
- 💪 **Production-ready design:** Modular, maintainable, and scalable for large document QA workflows.


## Quickstart

For one document, the example is as follows:

```python
# export GEMINI_API_KEY=AI...
# uses litellm under the hood
from pageindex_open import *

PDF_FILE = "/path/to/file/2023-annual-report-truncated.pdf"
QUERY = "what about financial stability?"


pio = PIO(PDF_FILE)
pio.build_index() 

answer = pio.query(QUERY, top_k=2)
print(answer)
```

## Application 

This works for structured documents and you applies to sectors like finance and legal

## API

Specify more

```py
pio = PIO(PDF_FILE, model_name="modelprovider/model-name", llm_client=litellm_client_if_any)
```

Load index

```py
from pageindex_open import *

PDF_FILE = "/path/to/file/2023-annual-report-truncated.pdf"
QUERY = "what about financial stability?"


pio = PIO(PDF_FILE)
pio.load_index("/path/mdfile.md", "/path/file.tree.json") # files that were created using build_index
```

## Roadmap

- [ ] Multi-document
- [ ] Document processing backend
- [ ] Save config options
- [ ] Add chat with docs feature
