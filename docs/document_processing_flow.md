# Document Processing Flow

This sequence diagram outlines the steps taken when the `python main.py process` command is executed.

```mermaid
sequenceDiagram
    actor Operator
    participant CLI [main.py process]
    participant DocProcessor [DocumentProcessor]
    participant FileLoader [Loader (LangChain/Docling/Mistral)]
    participant TextSplitter
    participant OpenAIEmbed
    participant AsyncSession [DB Session]
    participant PostgresDB

    Operator->>+CLI: Execute `process --dir <dir>`
    CLI->>+DocProcessor: process_directory(directory_path)
    DocProcessor->>DocProcessor: List files in directory
    loop For Each File
        DocProcessor->>+AsyncSession: check_duplicate_document(checksum)
        AsyncSession->>+PostgresDB: Query documents table by checksum
        PostgresDB-->>-AsyncSession: Existing document? (Yes/No)
        alt Checksum Exists
            AsyncSession-->>-DocProcessor: Return existing doc info
            DocProcessor->>DocProcessor: Log Skipping File
        else Checksum Does Not Exist
            AsyncSession-->>-DocProcessor: Return None
            DocProcessor->>+FileLoader: load(file_path)
            Note over FileLoader: Uses specific loader based on config/file type
            alt Mistral Loader Used
               FileLoader->>MistralOCR: Request OCR
               MistralOCR-->>FileLoader: OCR Text/Data
            end
            FileLoader-->>-DocProcessor: Return List[Document]
            DocProcessor->>+TextSplitter: split_documents(docs)
            TextSplitter-->>-DocProcessor: Return chunks (List[Document])
            DocProcessor->>+OpenAIEmbed: aembed_documents(chunk_texts)
            OpenAIEmbed-->>-DocProcessor: Return embeddings (List[vector])
            DocProcessor->>+AsyncSession: Store document metadata & chunks
            AsyncSession->>+PostgresDB: INSERT into documents & document_chunks tables
            PostgresDB-->>-AsyncSession: Confirm insert
            AsyncSession-->>-DocProcessor: Confirm storage
            DocProcessor->>DocProcessor: Log File Processed
        end
    end
    DocProcessor-->>-CLI: Return processing summary
    CLI-->>Operator: Display summary
```

**Explanation:**

1.  **Execution:** An operator runs the `process` command via the `main.py` CLI.
2.  **Processor Initialization:** The `DocumentProcessor` class from `document_rag_loader.py` is instantiated.
3.  **Directory Scan:** The processor lists files in the target directory.
4.  **File Loop:** For each file:
    *   **Checksum Calculation:** The SHA-256 checksum of the file content is calculated.
    *   **Duplicate Check:** The processor queries the **PostgreSQL database** (via an `AsyncSession`) to see if a document with the same checksum already exists.
    *   **Skip or Process:**
        *   If the checksum exists, the file is logged as skipped.
        *   If the checksum is new:
            *   **Loading:** The appropriate **File Loader** (selected based on configuration and file type, potentially using **Mistral OCR**) reads the file content.
            *   **Splitting:** The loaded content is passed to a **Text Splitter** (e.g., `RecursiveCharacterTextSplitter`) to create smaller chunks.
            *   **Embedding:** The text of each chunk is sent to the **OpenAI Embedding API** to get vector embeddings.
            *   **Storage:** The processor opens a database session and inserts the document metadata (filename, checksum) into the `documents` table and the chunk content, metadata, and vector embedding into the `document_chunks` table in **PostgreSQL**.
5.  **Summary:** After processing all files, a summary is returned to the CLI and displayed to the operator.
*   **Table Creation:** The `process` command first ensures necessary tables (`documents`, `document_chunks`) exist by calling `Base.metadata.create_all`. 