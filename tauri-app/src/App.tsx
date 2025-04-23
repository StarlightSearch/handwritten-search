import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";
import { openPath } from "@tauri-apps/plugin-opener";

interface SearchResult {
  file_path: string;
  text: string;
}

function App() {
  const [selectedFileName, setSelectedFileName] = useState("");
  const [fileContent, setFileContent] = useState<Uint8Array | null>(null);
  const [transcriptionResult, setTranscriptionResult] = useState<string | null>(
    null
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<
    Array<{ file_path: string; text: string }>
  >([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  async function openFile() {
    try {
      const filePath = await open({
        multiple: false,
      });

      if (filePath) {
        setSelectedFileName(filePath);
        const content = await readFile(filePath as string);
        const uint8Array = new Uint8Array(content);
        setFileContent(uint8Array);
      }
    } catch (err) {
      console.error("Failed to open file:", err);
    }
  }

  async function transcribeImage() {
    if (fileContent) {
      setIsProcessing(true);
      try {
        const result = await invoke("embed_ocr_text", {
          filePath: selectedFileName,
        });
        setTranscriptionResult(result as string);
        console.log("Embedding result:", result);
      } catch (err) {
        console.error("Processing failed:", err);
        setTranscriptionResult(null);
      } finally {
        setIsProcessing(false);
      }
    }
  }

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setIsSearching(true);
    try {
      const results = await invoke<SearchResult[]>("search_text", {
        query: searchQuery,
      });
      setSearchResults(
        results.map((result: SearchResult) => ({
          file_path: result.file_path,
          text: result.text,
        }))
      );
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setIsSearching(false);
    }
  }

  async function openResultFile(filePath: string) {
    try {
      console.log("Opening file:", filePath);
      await openPath(filePath);
    } catch (err) {
      console.error("Failed to open file:", err);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center text-gray-900 mb-12">
          Handwritten Search
        </h1>

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <div className="flex items-center justify-center gap-4 mb-6">
            <button
              onClick={openFile}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium shadow-sm"
            >
              Open File
            </button>
            <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded">
              {selectedFileName || "No file selected"}
            </span>
          </div>

          <button
            onClick={transcribeImage}
            disabled={!fileContent || isProcessing}
            className={`w-full px-6 py-3 rounded-lg font-medium transition-colors ${
              !fileContent || isProcessing
                ? "bg-gray-300 cursor-not-allowed"
                : "bg-green-600 hover:bg-green-700 text-white"
            }`}
          >
            {isProcessing ? "Processing..." : "Transcribe & Index"}
          </button>

          {transcriptionResult && (
            <div className="mt-8 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Transcription:
              </h3>
              <pre className="whitespace-pre-wrap text-gray-700 font-mono">
                {transcriptionResult}
              </pre>
            </div>
          )}
        </div>

        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">
            Search Documents
          </h3>
          <form onSubmit={handleSearch} className="mb-6">
            <div className="flex gap-2">
              <input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Enter search query..."
                required
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                type="submit"
                disabled={isSearching}
                className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                  isSearching
                    ? "bg-gray-300 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
              >
                {isSearching ? "Searching..." : "Search"}
              </button>
            </div>
          </form>

          {searchResults.length > 0 && (
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-900">
                Search Results:
              </h4>
              {searchResults.map((result, index) => (
                <div
                  key={index}
                  onClick={() => openResultFile(result.file_path)}
                  className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
                >
                  <span className="text-sm text-gray-500 block mb-2">
                    {result.file_path}
                  </span>
                  <p className="text-gray-700">{result.text}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <a
          href="/slides/Slide1_1"
          className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium shadow-sm text-center my-2"
        >
          Next
        </a>
      </div>
    </main>
  );
}

export default App;
