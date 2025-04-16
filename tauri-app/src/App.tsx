import { useState } from "react";
import reactLogo from "./assets/react.svg";
import { invoke } from "@tauri-apps/api/core";

function App() {
  const [greetMsg, setGreetMsg] = useState("");
  const [name, setName] = useState("");

  async function greet() {
    // Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
    setGreetMsg(await invoke("greet", { name }));
  }

  return (
    <main className="min-h-screen bg-gray-100 py-8 px-4 mx-auto max-w-7xl">
      <h1 className="text-3xl font-bold text-center mb-8">Welcome to Tauri + React</h1>

      <div className="flex justify-center gap-8 mb-8">
        <a href="https://vitejs.dev" target="_blank" className="hover:opacity-75 transition-opacity">
          <img src="/vite.svg" className="h-16" alt="Vite logo" />
        </a>
        <a href="https://tauri.app" target="_blank" className="hover:opacity-75 transition-opacity">
          <img src="/tauri.svg" className="h-16" alt="Tauri logo" />
        </a>
        <a href="https://reactjs.org" target="_blank" className="hover:opacity-75 transition-opacity">
          <img src={reactLogo} className="h-16" alt="React logo" />
        </a>
      </div>
      <p className="text-center text-gray-600 mb-8">Click on the Tauri, Vite, and React logos to learn more.</p>

      <form
        className="flex flex-col items-center gap-4 max-w-md mx-auto"
        onSubmit={(e) => {
          e.preventDefault();
          greet();
        }}
      >
        <input
          id="greet-input"
          onChange={(e) => setName(e.currentTarget.value)}
          placeholder="Enter a name..."
          className="w-full px-4 py-2 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <button 
          type="submit"
          className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Greet
        </button>
      </form>
      <p className="text-center mt-4 text-gray-700">{greetMsg}</p>
    </main>
  );
}

export default App;
