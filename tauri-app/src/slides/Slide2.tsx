import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export default function Slide2() {
  const [selectedFile, setSelectedFile] = useState<string>("");
  const [ocrText, setOcrText] = useState<string>("");

  const handleFileSelect = async () => {
    try {
      const filePath = await open({
        multiple: false,
      });

      if (filePath) {
        setSelectedFile(filePath);
        // Invoke OCR with the file path
        try {
          const text = await invoke("ocr_text", { filePath });
          setOcrText(text as string);
        } catch (err) {
          console.error("Failed to perform OCR:", err);
          setOcrText("Error performing OCR");
        }
      }
    } catch (err) {
      console.error("Failed to open file:", err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center text-gray-900 mb-12">
          File Reader - Slide 2
        </h1>
        <div className="bg-white rounded-xl shadow-lg p-6">
          <div className="flex items-center justify-center gap-4 mb-6">
            <button
              onClick={handleFileSelect}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium shadow-sm"
            >
              Select File
            </button>
            <span className="text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded">
              {selectedFile || "No file selected"}
            </span>
          </div>
          {ocrText && (
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h2 className="text-lg font-semibold text-gray-800 mb-2">Extracted Text:</h2>
              <pre className="whitespace-pre-wrap text-gray-700">{ocrText}</pre>
            </div>
          )}
        </div>
      </div>

      <a 
        href="/" 
        className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium shadow-sm text-center my-2"
      >
        Next
      </a>
    </div>
  );
}
