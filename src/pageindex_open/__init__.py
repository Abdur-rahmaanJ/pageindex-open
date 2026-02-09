import json
import re
from pathlib import Path
from typing import List, Dict, Any
import litellm
from pymupdf4llm import to_markdown


def pdf_to_markdown(pdf_path: str) -> str:
    return to_markdown(pdf_path)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")

def build_md_structure(md: str, doc_name: str) -> Dict[str, Any]:
    lines = md.splitlines()

    structure = []
    stack = []
    node_counter = 0

    def new_node(title, start, level):
        nonlocal node_counter
        node = {
            "title": title,
            "start_index": start,
            "end_index": None,
            "node_id": f"{node_counter:04d}",
            "level": level,
            "nodes": []
        }
        node_counter += 1
        return node

    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue

        level = len(m.group(1))
        title = m.group(2).strip()
        node = new_node(title, i + 1, level)

        while stack and stack[-1]["level"] >= level:
            finished = stack.pop()
            finished["end_index"] = i

        if stack:
            stack[-1]["nodes"].append(node)
        else:
            structure.append(node)

        stack.append(node)

    for n in stack:
        n["end_index"] = len(lines)

    def clean(node):
        node.pop("level")
        if not node["nodes"]:
            node.pop("nodes")
        else:
            for c in node["nodes"]:
                clean(c)

    for n in structure:
        clean(n)

    return {
        "doc_name": doc_name,
        "structure": structure,
        "md_lines": lines
    }


def save_markdown(md: str, pdf_path: str) -> Path:
    md_path = Path(pdf_path).with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    return md_path

def save_tree(index: dict, pdf_path: str) -> Path:
    tree_path = Path(pdf_path).with_suffix(".tree.json")
    payload = {
        "doc_name": index["doc_name"],
        "structure": index["structure"]
    }
    tree_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return tree_path



def gemini_route_top_k(query: str, index: Dict[str, Any], top_k: int = 3) -> List[str]:
    prompt = f"""
You are a document router.

Given a document structure with section titles and ranges, select the {top_k} most relevant section IDs for this question.

Rules:
- Return JSON only
- One key: "node_ids" (list of {top_k} IDs)
- No explanations

Question:
{query}

Structure:
{json.dumps(index["structure"], indent=2)}

Return:
{{"node_ids": ["0003", "0005", "0001"]}}
"""
    resp = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    content = resp["choices"][0]["message"]["content"]
    if isinstance(content, str):
        content = json.loads(content)
    return content["node_ids"]


def find_node(structure: List[Dict], node_id: str) -> Dict:
    for n in structure:
        if n["node_id"] == node_id:
            return n
        if "nodes" in n:
            r = find_node(n["nodes"], node_id)
            if r:
                return r
    return None

def extract_section_text(md_lines: List[str], node: Dict) -> str:
    return "\n".join(md_lines[node["start_index"]:node["end_index"]]).strip()


def gemini_answer(query: str, node_text: str) -> str:
    prompt = f"""
You are an expert assistant.

Use ONLY the following context to answer the question.
Do not hallucinate beyond this context.

Context:
{node_text[:3000]}

Question:
{query}

Answer concisely and clearly.
"""
    resp = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp["choices"][0]["message"]["content"].strip()





class PIO:

    def __init__(
        self, 
        pdf_path: str, 
        llm_client=None, 
        model_name: str = "gemini/gemini-2.5-flash"
    ):
        """
        Args:
            pdf_path: Path to PDF
            llm_client: Optional custom LLM client (defaults to litellm)
            model_name: Model name to use with LLM
        """
        self.pdf_path = Path(pdf_path)
        self.md = ""
        self.md_lines: List[str] = []
        self.index: Dict = {}
        self.llm_client = llm_client or litellm  # default to litellm
        self.model_name = model_name

    def build_index(self, save_files: bool = True):
        """Convert PDF → Markdown → build tree, optionally save files."""
        self.md = pdf_to_markdown(str(self.pdf_path))
        self.md_lines = self.md.splitlines()
        self.index = build_md_structure(self.md, self.pdf_path.name)

        if save_files:
            save_markdown(self.md, self.pdf_path)
            save_tree(self.index, self.pdf_path)

        return self.index

    def query(self, query: str, top_k: int = 1) -> str:
        """
        Query the document using Gemini.
        Returns answer using top_k most relevant nodes.
        """
        if not self.index:
            raise ValueError("Index not built yet. Call build_index() first.")

        node_ids = self._route_top_k(query, top_k=top_k)

        combined_texts = []
        for node_id in node_ids:
            node = find_node(self.index["structure"], node_id)
            text = extract_section_text(self.md_lines, node)
            combined_texts.append(text)

        full_context = "\n\n".join(combined_texts)
        answer = self._answer(query, full_context)
        return answer

    def _route_top_k(self, query: str, top_k: int) -> List[str]:
        """Internal method to route query to top_k nodes using LLM."""
        prompt = f"""
You are a document router.

Given a document structure with section titles and ranges, select the {top_k} most relevant section IDs for this question.

Rules:
- Return JSON only
- One key: "node_ids" (list of {top_k} IDs)
- No explanations

Question:
{query}

Structure:
{json.dumps(self.index["structure"], indent=2)}

Return:
{{"node_ids": ["0003", "0005", "0001"]}}
"""
        resp = self.llm_client.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = resp["choices"][0]["message"]["content"]
        if isinstance(content, str):
            content = json.loads(content)
        return content["node_ids"]

    def _answer(self, query: str, node_text: str) -> str:
        """Internal method to get answer using LLM."""
        prompt = f"""
You are an expert assistant.

Use ONLY the following context to answer the question.
Do not hallucinate beyond this context.

Context:
{node_text[:3000]}

Question:
{query}

Answer concisely and clearly.
"""
        resp = self.llm_client.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp["choices"][0]["message"]["content"].strip()

    def save_index(self, md_path: str = None, tree_path: str = None):
        """Save markdown and tree to disk"""
        md_path = md_path or str(self.pdf_path.with_suffix(".md"))
        tree_path = tree_path or str(self.pdf_path.with_suffix(".tree.json"))
        save_markdown(self.md, md_path)
        save_tree(self.index, tree_path)

    def load_index(self, md_path: str, tree_path: str):
        """Load markdown and tree from disk"""
        self.md = Path(md_path).read_text(encoding="utf-8")
        self.md_lines = self.md.splitlines()
        self.index = json.loads(Path(tree_path).read_text(encoding="utf-8"))

