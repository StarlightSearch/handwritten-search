import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

export default function Slide1_2() {
  const [selectedFile, setSelectedFile] = useState<string>("");

  const handleFileSelect = async () => {
    try {
      const filePath = await open({
        multiple: false,
      });

      if (filePath) {
        setSelectedFile(filePath);
        // Invoke greet with the file name
        try {
          const greeting = await invoke("greet", { name: filePath });
          console.log("Greeting result:", greeting);
        } catch (err) {
          console.error("Failed to greet:", err);
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
          File Reader - Slide 1_2
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
        </div>
      </div>

      <a 
        href="/slides/Slide2" 
        className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium shadow-sm text-center my-2"
      >
        Next
      </a>
    </div>
  );
}
