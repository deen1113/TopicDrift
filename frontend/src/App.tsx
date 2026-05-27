import { Routes, Route } from "react-router-dom";

export default function App() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <Routes>
        <Route path="/" element={<Home />} />
      </Routes>
    </div>
  );
}

function Home() {
  return (
    <main className="mx-auto max-w-3xl p-8">
      <h1 className="text-3xl font-bold">TopicDrift</h1>
      <p className="mt-2 text-muted-foreground">
        Topic drift analysis for long-running SE conferences.
      </p>
    </main>
  );
}
