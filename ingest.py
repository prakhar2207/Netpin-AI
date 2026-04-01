import requests
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from playwright.sync_api import sync_playwright

BASE_URL = "https://netpin.io/docs/"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ---------------------------
# STEP 1: SCRAPE LINKS
# ---------------------------
def get_links():
    links = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        print("🌐 Loading page with browser...")
        page.goto(BASE_URL, timeout=60000)

        # wait for JS to load content
        page.wait_for_timeout(5000)

        anchors = page.query_selector_all("a")

        for a in anchors:
            href = a.get_attribute("href")

            if href and "/docs" in href:
                if href.startswith("http"):
                    links.add(href)
                else:
                    links.add("https://netpin.io" + href)

        browser.close()

    links = list(links)
    print(f"✅ Found {len(links)} links via Playwright")

    return links


# ---------------------------
# STEP 2: EXTRACT TEXT
# ---------------------------
def extract_text(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)

        if res.status_code != 200:
            return ""

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = " ".join(text.split())

        return text

    except Exception as e:
        print(f"❌ Error extracting {url}: {e}")
        return ""


# ---------------------------
# STEP 3: CREATE DOCUMENTS
# ---------------------------
def create_docs(urls):
    docs = []

    for url in urls:
        text = extract_text(url)

        if text and len(text) > 100:  # avoid empty/useless pages
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": url}
                )
            )
            print(f"✅ Added doc: {url} ({len(text)} chars)")

    print(f"📄 Total valid docs: {len(docs)}")
    return docs


# ---------------------------
# STEP 4: CHUNKING
# ---------------------------
def chunk_docs(docs):
    if not docs:
        raise ValueError("❌ No documents found. Scraping failed.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("❌ No chunks created.")

    print(f"✂️ Total chunks: {len(chunks)}")
    return chunks


# ---------------------------
# STEP 5: EMBEDDINGS
# ---------------------------
def get_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cuda"}  # change to "cpu" if needed
        )

        # sanity check
        test = embeddings.embed_query("hello world")
        print(f"🧠 Embedding dimension: {len(test)}")

        return embeddings

    except Exception as e:
        print(f"❌ Embedding error: {e}")
        raise


# ---------------------------
# STEP 6: VECTOR STORE (FIXED)
# ---------------------------
def create_vectorstore(chunks, embeddings):
    print("⚡ Building FAISS index...")

    # 🔥 IMPORTANT FIX: let LangChain handle index
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("✅ FAISS index created")
    return vs


# ---------------------------
# STEP 7: SAVE DB
# ---------------------------
def save_db(vs):
    try:
        vs.save_local("db")
        print("💾 Database saved successfully")

    except Exception as e:
        print(f"❌ Error saving DB: {e}")


# ---------------------------
# RUN PIPELINE
# ---------------------------
def main():
    print("🔍 Fetching links...")
    urls = get_links()

    if not urls:
        raise ValueError("❌ No URLs found")

    print("📄 Creating documents...")
    docs = create_docs(urls)

    print("✂️ Chunking...")
    chunks = chunk_docs(docs)

    print("🧠 Loading embeddings...")
    embeddings = get_embeddings()

    print("⚡ Creating vector store...")
    vs = create_vectorstore(chunks, embeddings)

    print("💾 Saving database...")
    save_db(vs)

    print("🎉 Done!")


if __name__ == "__main__":
    main()