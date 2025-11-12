import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Home } from "lucide-react";
import HomePage from "./pages/HomePage";
import LoginPage  from "./pages/LoginPage";
import HistoryPage from "./pages/HistoryPage";

export default function App() {
  return (
      <BrowserRouter>
        <Routes>
            <Route path="/" element={<HomePage/>} />
           <Route path="/login" element={<LoginPage/>} />
            <Route path="/summaries/:id" element={<HistoryPage/>} />
        </Routes>
      </BrowserRouter>
  );
}
